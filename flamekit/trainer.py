import os
from pathlib import Path
from datetime import datetime

from flamekit.devices import to_device 
from flamekit.callbacks import Callback
from flamekit.pbars import TQDMProgressBar
from flamekit.plotting import plot_curve_groups

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class TorchTrainer:
    """ 
    Minimalistic class for performing training and evaluation over a PyTorch model. It is a 
    helper class designed to eliminate boilerplate code needed for looping over the dataset,
    logging metrics, and plotting results. Each critical part of the training and evaluation
    phases is implemented in a different compartmentalized function, which can be overridden
    to cater to specific use cases. It is intended to be lightweight, fast, and highly
    customizable.

    This class facilitates various aspects of the training process, including checkpoint saving,
    callback hooks, metric logging, and more. The trainer allows users to train models, resume
    training processes from saved checkpoints, and evaluate performance through logging and
    plotting of metrics.
    """
    MIN_MODE = 'min'
    MAX_MODE = 'max'
    TRAIN = 'train'
    VAL = 'val'
    PREDICT = 'predict'
    
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
        self.results_ncols = 3
        self.results_path = None
        self.train_color = '#1f77b4' # Blue
        self.val_color = '#ff7f0e'   # Orange
        self.extension = 'pt'
        
    def compile(self, optimizer:optim.Optimizer, criterion:nn.Module=None):
        self.optimizer = optimizer
        self.criterion = criterion
        
    def is_compiled(self):
        return hasattr(self, 'optimizer') and self.optimizer is not None
        
    def __save_model(self, checkpoint_dir:'str | Path', monitor_metric, mode, save_best=True, prefix=None):
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        prefix = prefix or 'ckp'
        metric_epoch_array = self.history[monitor_metric]
        
        if save_best:
            index = np.argmin(metric_epoch_array) if mode == self.MIN_MODE else np.argmax(metric_epoch_array) 
            if index != len(metric_epoch_array) - 1: return
            self.best_index = index
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
        else: 
            index = -1
            if self.last_model_path is not None:
                os.remove(self.last_model_path)
                
        epoch = self.current_epoch + 1
        assert epoch == len(metric_epoch_array)
        metric = round(metric_epoch_array[index], ndigits=4)
        
        suffix = 'best' if save_best else 'last'
        name = f'{prefix}_{monitor_metric.replace("_", "-")}_{metric}_{epoch}_{suffix}.{self.extension}'
        path_to_save = checkpoint_dir/name
        if save_best: 
            self.best_model_path = path_to_save
            msg = f"[INFO] Saving best checkpoint, regarding '{monitor_metric}' metric -- mode='{mode}' ({path_to_save})"
            print(msg)
        else:
            self.last_model_path = path_to_save
        
        data_to_save = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'monitor': monitor_metric,
            'date': datetime.now().isoformat()
        }
        self.save_model(path_to_save, data_to_save)
        
    def save_model(self, path_to_save, data_to_save:dict):
        torch.save(data_to_save, path_to_save)
    
    def __save_results(self, dest_path):    
        epoch = self.current_epoch
        keys, vals = list(self.history.keys()), [v[epoch] for v in self.history.values()]
        n = len(self.history) + 1  # number of cols
        s = "" if dest_path.exists() else (("%15s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(dest_path, "a") as f:
            f.write(s + ("%15.5g," * n % tuple([epoch + 1] + vals)).rstrip(",") + "\n")
      
    def load(self, model_path:Path):
        self.checkpoint = torch.load(model_path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.history = self.checkpoint['history']
        self.current_epoch = self.checkpoint['epoch']
        assert len(self.history['loss']) == self.current_epoch
        print(f"[OK] Checkpoint '{model_path}' has been loaded successfully")
        
    def load_best(self):
        if self.best_model_path is None:
            raise RuntimeError("No best model has been saved, fit() method has not been called")
        self.load(self.best_model_path) 
        
    def load_last(self):
        if self.last_model_path is None:
            raise RuntimeError("No last model has been saved, fit() method has not been called")
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
    
    def log(self, metrics:list[tuple], average=True):
        """ Logs values into a dictionary that is averaged on each step """
        for (k, v) in metrics:
            if self.step_logs.get(k, False):
                updated_count = self.step_logs[k][1] + 1
                previous_avg_value = self.step_logs[k][0]
                updated_avg = v if not average else self.step_average(v, previous_avg_value, updated_count)
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
        if not self.is_compiled():
            raise RuntimeError("You have to compile the trainer before calling 'fit'")
        self.training = True
        self.terminate = False
        self.num_training_batches = len(train_loader)
        self.monitor = monitor
        if 'val_' in monitor:
            self.monitor = monitor if validation_loader is not None else monitor.replace('val_', '')
        
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
                    
            for c in callbacks: c.on_fit_epoch_end(self, self.model)
            self.__record_and_clear_step_logs()
            
            if dest_path is not None:
                if not isinstance(dest_path, Path):
                    dest_path = Path(dest_path)
                if save_best:
                    self.__save_model(dest_path, self.monitor, mode, save_best=True, prefix=prefix)
                self.__save_model(dest_path, self.monitor, mode, save_best=False, prefix=prefix)
                self.__save_results(dest_path/'results.csv')
            self.current_epoch += 1
            if self.terminate: break
        
        self.training = False
        for c in callbacks: c.on_fit_end(self, self.model)
        if dest_path is not None: 
            fig, _ = self.__plot_results(dest_path/'results.png')
            plt.close(fig)
        print("[END] Training finished")
    
        return self.history
    
    def loss_step(self, outputs, labels, stage:str=TRAIN) -> torch.Tensor:
        """ 
        Computes the loss for a given set of model predictions and labels and processes the criterion output,
        which can be either a tensor or a scalar and logs the result. 
        
        This function can be overwritten to change the default behaviour of how the loss is computed and logged
        in each stage.
        
        Returns:
            torch.Tensor: The computed loss value on which .backward() will be called
        """
        if self.criterion is None:
            raise RuntimeError("You have to provide a loss function in trainer.compile() " + 
                               "or override trainer.loss_step() method")
       
        loss = self.criterion(outputs, labels)

        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Criterion output type '{type(loss)}' not supported, " + 
                             "override trainer.loss_step() to make it compatible with your implementation")
        elif loss.numel() > 1:
            raise ValueError("Loss must be scalar")
        
        name = 'loss' if stage != self.VAL else 'val_loss'
        self.log([(name, loss.item())])
        return loss
    
    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def training_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        outputs = self.model(inputs)
        step_loss = self.loss_step(outputs, labels, self.TRAIN)
        return outputs, step_loss
    
    def __train(self, train_loader, callbacks:list[Callback]=None) -> tuple:
        """ Train model for one epoch """
        self.model.train()
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = to_device(data, self.device)
            batch = (inputs, labels)
            for c in callbacks: c.on_train_batch_start(self, self.model, batch, batch_idx)
            outputs, step_loss = self.training_step(batch, batch_idx)
            self.optimizer_step(step_loss, self.optimizer)
            for c in callbacks: c.on_train_batch_end(self, self.model, outputs, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        self.loss_step(outputs, labels, self.VAL)
        return outputs
    
    def __validate(self, validation_loader, callbacks:list[Callback]=[]) -> tuple:
        """ Validate model with validation data """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                inputs, labels = to_device(data, self.device)
                batch = (inputs, labels)
                for c in callbacks: c.on_validation_batch_start(self, self.model, batch, batch_idx)
                outputs = self.validation_step(batch, batch_idx)
                for c in callbacks: c.on_validation_batch_end(self, self.model, outputs, batch, batch_idx) 
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        self.loss_step(outputs, labels, self.PREDICT)
        return outputs
    
    def predict(self, test_loader, callbacks:list=[TQDMProgressBar()]):
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
                outputs = self.predict_step(batch, batch_idx)
                for c in callbacks: c.on_predict_batch_end(self, self.model, outputs, batch, batch_idx)
            
            for c in callbacks: c.on_predict_epoch_end(self, self.model)
            self.__record_and_clear_step_logs()
        
        for c in callbacks: c.on_predict_end(self, self.model)
    
    def plot(self, metrics:list=None, add_label_text=False, dest_path=None, colors:list=None):
        """ 
        Plots metrics from history and optionally saves the plot to a file.
        
        Args:
            metrics (list, optional): List of metric names to plot. If not provided, all available 
                metrics will be plotted.
            add_label_text (bool, optional): Whether to add label text to the plot instead of using 
                the legend. Defaults to False.
            dest_path (str or Path, optional): Path to save the plot. If not provided, the plot will 
                only be displayed.
            colors (list, optional): List of colors to use for plotting. If not provided, default 
                matplotlib colors will be used. Can be a list of lists, containing the colors for each
                metric group or a single list with same colors to use for all metrics.
        """
        if not metrics:
            keys = self.history.keys()
        else:
            keys = []
            for k in metrics:
                if k not in self.history:
                    if 'val_'+k not in self.history:
                        raise ValueError(f"Metric '{k}' not found in history")
                elif k not in keys: 
                    keys.append(k)
                if 'val_' not in k and 'val_'+k in self.history and 'val_'+k not in keys:
                    keys.append('val_'+k)
        self.__plot_results(dest_path, keys=keys, colors=colors, add_label_text=add_label_text)
        plt.tight_layout()
        plt.show()
        
    def __plot_results(self, dest_path=None, keys:list=None, colors=None, add_label_text=False):
        # Create groups of metrics (plot training and val metrics together)
        groups = []
        dic = self.history
        if keys is not None:
            dic = {k:v for (k, v) in dic.items() if k in keys}
        metrics = {k:v for (k, v) in dic.items() if not k.startswith('val_')}
        val_metrics = {k:v for (k, v) in dic.items() if k.startswith('val_')}
        for (k, v) in metrics.items():
            val_name = 'val_'+k
            if val_name in val_metrics:
                val_v = val_metrics.pop(val_name)
                groups.append({k:v, val_name:val_v})
            else:
                groups.append({k:v})
        for (k, v) in val_metrics.items(): # Remaining val metrics
            groups.append({k:v})
        # Define curve colors
        if colors is None:
            colors = []
            for group in groups:
                group_colors = []
                for k in group.keys():
                    group_colors.append(self.val_color if 'val_' in k else self.train_color)
                colors.append(group_colors)
        # Plot curve groups
        def customize_axis(idx:int, ax:plt.Axes, group:dict, colors):
            title = list(group.keys())[0].replace('val_', '').replace('_', ' ')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            
        fig, axes = plot_curve_groups(groups, ncols=self.results_ncols, colors=colors,
                          add_label_text=add_label_text, dest_path=dest_path, callback=customize_axis)
        return fig, axes