import os
from pathlib import Path

from flamekit.devices import to_device 
from flamekit.callbacks import Callback
from flamekit.pbars import TQDMProgressBar

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


MIN_MODE = 'min'
MAX_MODE = 'max'
             
class TorchTrainer:
    """ 
    Class for performing training and evaluation over a PyTorch model.
    
    This class facilitates various aspects of the training process, including checkpoint saving,
    callback hooks, metric logging, and more. The trainer allows users to train models and resume
    training processes from saved checkpoints, as well as evaluating performance through logging 
    and plotting of metrics. Additionally, it supports inference and evaluation of the predicted data.
    """
    
    def __init__(self, model:nn.Module, device) -> None:
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.current_epoch = 0
        self.terminate = False
        self.history = {} # For epoch metrics
        self.step_logs = {} # For step metrics (values are averaged on each step to save memory usage)
        self.best_model_path = None
        self.last_model_path = None
        self.training = False
        
    def compile(self, optimizer:optim.Optimizer, criterion:nn.Module=None):
        self.optimizer = optimizer
        self.criterion = criterion
        
    def __save_model(self, checkpoint_dir:'str | Path', monitor_metric, mode, save_best=True, prefix=None):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        metric_epoch_array = self.history[monitor_metric]
        
        if save_best:
            index = np.argmin(metric_epoch_array) if mode == MIN_MODE else np.argmax(metric_epoch_array) 
            if index != len(metric_epoch_array) - 1: return
            self.best_index = index
            for checkpoint_file in checkpoint_dir.glob(f'{prefix}*'):
                os.remove(checkpoint_file)
        else: index = -1
        
        epoch = self.current_epoch + 1
        assert epoch == len(metric_epoch_array)
        metric = round(metric_epoch_array[index], ndigits=4)
        
        suffix = 'best' if save_best else 'last'
        name = f'{prefix}_{monitor_metric.replace("_", "-")}_{metric}_{epoch}_{suffix}.tar'
        path_to_save = checkpoint_dir/name
        self.last_model_path = path_to_save
        msg = f"[INFO] Saving last checkpoint of the training to '{path_to_save}'"
        if save_best: 
            self.best_model_path = path_to_save
            msg = f"[INFO] Saving best checkpoint, regarding '{monitor_metric}' metric -- mode='{mode}' ({path_to_save})"
        print(msg)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'monitor': monitor_metric
        }, path_to_save)
        
    def load(self, model_path:Path):
        self.checkpoint = torch.load(model_path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.history = self.checkpoint['history']
        self.current_epoch = self.checkpoint['epoch']
        assert len(self.history['loss']) == self.current_epoch
        if self.best_model_path is None: 
            self.best_model_path = model_path
            self.last_model_path = model_path
        print(f"[OK] Checkpoint '{model_path}' has been loaded successfully")
        
    def load_best(self):
        self.load(self.best_model_path) 
        
    def load_last(self):
        self.load(self.last_model_path)
    
    def __record_in_history(self, metrics:list[tuple]):
        for (k,v) in metrics:
            assert type(v) is float or type(v) is np.float_, f"Metric value recorded is not 'float' but '{type(v)}'"
            if self.history.get(k, False):
                if len(self.history[k]) == self.current_epoch:
                    self.history[k].append(v)
                elif len(self.history[k]) - 1 == self.current_epoch:
                    self.history[k][self.current_epoch] = v
                else:
                    raise RuntimeError("Current epoch and history dictionary are not synchronized")
            else:
                nan_array = [np.nan]*self.current_epoch
                self.history[k] = nan_array + [v]
                
    def __history_sanity_check(self):
        for (k,v) in self.history.items():
            if len(v) == self.current_epoch:
                self.history[k].append(np.nan)
            elif len(v) < self.current_epoch:
                raise RuntimeError("Sanity Check: Current epoch and history dictionary are not synchronized")   
    
    def log(self, metrics:list[tuple]):
        """ Logs values into a dictionary that is averaged on each step """
        for (k, v) in metrics:
            if self.step_logs.get(k, False):
                updated_count = self.step_logs[k][1] + 1
                previous_avg_value = self.step_logs[k][0]
                updated_avg = self.step_average(v, previous_avg_value, updated_count)
                self.step_logs[k] = (updated_avg, updated_count)
            else:
                self.step_logs[k] = (v, 1)
                  
    def step_average(self, new_value, previous_avg_value, updated_count):
        """ 
        Computes standard average by default, this function can be overwritten
        to change this behaviour
        """
        return (previous_avg_value * (updated_count - 1) + new_value) / updated_count
    
    def __record_and_clear_step_logs(self):
        if self.training:
            self.__record_in_history(self.get_step_metrics())
            self.__history_sanity_check()
        self.step_logs = {}       
        
    def get_step_metrics(self) -> list[tuple]:
        """ Get current step metrics """
        values = [v[0] for v in self.step_logs.values()]
        return list(zip(self.step_logs.keys(), values))  
    
    def fit(self, train_loader, epochs, validation_loader=None, dest_path:'str | Path'=None, prefix=None,
                save_best=True, monitor='val_loss', mode='min', callbacks:list[Callback]=[TQDMProgressBar()]):
        self.training = True
        self.terminate = False
        self.num_training_batches = len(train_loader)
        if monitor == 'val_loss':
            self.monitor = 'val_loss' if validation_loader is not None else 'loss'
        
        self.max_epochs = self.current_epoch + epochs
        
        for c in callbacks: c.on_fit_start(self, self.model)
        
        for ei in range(epochs):
            for c in callbacks: c.on_train_epoch_start(self, self.model)
            self.__train(train_loader, callbacks=callbacks)
            for c in callbacks: c.on_train_epoch_end(self, self.model)

            if validation_loader is not None:
                for c in callbacks: c.on_validation_epoch_start(self, self.model)
                self.__validate(validation_loader, callbacks=callbacks)
                for c in callbacks: c.on_validation_epoch_end(self, self.model)
                    
            self.__record_and_clear_step_logs()
            for c in callbacks: c.on_fit_epoch_end(self, self.model)
            
            if dest_path is not None:
                save_best = save_best and (ei < epochs -1)
                self.__save_model(dest_path, self.monitor, mode, save_best=save_best, prefix=prefix)
            
            self.current_epoch += 1
            if self.terminate: break
        
        self.training = False
        for c in callbacks: c.on_fit_end(self, self.model)
    
        return self.history
    
    def __process_batch_loss(self, loss:'torch.Tensor | list[tuple] | tuple[tuple]', prefix=None) -> tuple:
        prefix = '' if prefix is None else prefix+"_"
        if isinstance(loss, tuple) or isinstance(loss, list):
            loss_dict_ = dict(loss) 
            loss_names_ = [prefix+l for l in loss_dict_.keys()]
            losses_ = torch.stack(list(loss_dict_.values()), dim=0)
            total_loss = torch.sum(losses_, dim=0)
            loss_names = [prefix+'loss'] + loss_names_
            loss_items = [total_loss.item()] + [l.item() for l in losses_]
        elif isinstance(loss, torch.Tensor):
            total_loss = loss
            loss_names = [prefix+'loss']
            loss_items = [loss.item()]
        else:
            raise RuntimeError(f"Criterion output type '{type(loss)}' not supported")

        return total_loss, loss_names, loss_items
    
    def training_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        outputs = self.model(inputs)
        step_loss = self.criterion(outputs, labels)
        return outputs, step_loss
    
    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def __train(self, train_loader, callbacks:list[Callback]=None) -> tuple:
        """ Train model for one epoch """
        self.model.train()
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = to_device(data, self.device)
            batch = (inputs, labels)

            for c in callbacks: c.on_train_batch_start(self, self.model, batch, batch_idx)
            
            outputs, step_loss = self.training_step(batch, batch_idx)
            step_loss, loss_names, loss_items = self.__process_batch_loss(step_loss)
            self.optimizer_step(step_loss, self.optimizer)
            
            step_metrics = list(zip(loss_names, loss_items))
            self.log(step_metrics)
            
            for c in callbacks: c.on_train_batch_end(self, self.model, outputs, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        step_loss = self.criterion(outputs, labels)
        return outputs, step_loss
    
    def __validate(self, validation_loader, callbacks:list[Callback]=[]) -> tuple:
        """ Validate model with validation data """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                inputs, labels = to_device(data, self.device)
                batch = (inputs, labels)
                
                for c in callbacks: c.on_validation_batch_start(self, self.model, batch, batch_idx)
                
                outputs, step_loss = self.validation_step(batch, batch_idx)
                step_loss, loss_names, loss_items = self.__process_batch_loss(step_loss, 'val')
                
                step_metrics = list(zip(loss_names, loss_items))
                self.log(step_metrics)
                
                for c in callbacks: c.on_validation_batch_end(self, self.model, outputs, batch, batch_idx) 
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        step_loss = self.criterion(outputs, labels)
        return outputs, step_loss
    
    def predict(self, test_loader, callbacks:list=[TQDMProgressBar()]):
        """ Predict samples and return input data and predictions"""
        self.model.eval()
        self.num_predict_batches = len(test_loader)
        assert not self.model.training, "Model is in training mode"
        
        for c in callbacks: c.on_predict_start(self, self.model)
        
        with torch.no_grad():
            for c in callbacks: c.on_predict_epoch_start(self, self.model)
            for batch_idx, data in enumerate(test_loader):
                inputs, labels = to_device(data, self.device)
                batch = (inputs, labels)

                for c in callbacks: c.on_predict_batch_start(self, self.model, batch, batch_idx)
                
                outputs, step_loss = self.predict_step(batch, batch_idx)
                step_loss, loss_names, loss_items = self.__process_batch_loss(step_loss)
                
                step_metrics = list(zip(loss_names, loss_items))
                self.log(step_metrics)
                
                for c in callbacks: c.on_predict_batch_end(self, self.model, outputs, batch, batch_idx)
            
            for c in callbacks: c.on_predict_epoch_end(self, self.model)
            self.__record_and_clear_step_logs()
        
        for c in callbacks: c.on_predict_end(self, self.model)
    
    def plot(self, metrics:list=['loss'], figsize=(15,5), style=False, ylim:tuple=(0,1)):
        if style: plt.style.use("ggplot")
        
        plt.figure(figsize=figsize)
        def plot_history_metric(metric):
            plt.plot(self.history[f"{metric}"])
            val_metric = f"val_{metric}"
            if val_metric in self.history: plt.plot(self.history[val_metric])
            plt.title('Train History')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend(['train', 'validation'], loc='upper left')
        
        for i, m in enumerate(metrics):
            plt.subplot(1,len(metrics), i+1)
            plot_history_metric(m)
        plt.show()
        

class AMPTrainer(TorchTrainer):
    """ TorchTrainer subclass that enables Automatic Mixed Precision (AMP) training. """
    
    def __init__(self, model, device, amp_dtype=torch.float16) -> None:
        super().__init__(model, device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.amp_dtype = amp_dtype
    
    def training_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        with torch.autocast(device_type=inputs.device.type, dtype=self.amp_dtype):
            outputs = self.model(inputs)
            step_loss = self.criterion(outputs, labels)
        return outputs, step_loss

    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()