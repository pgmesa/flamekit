
import torch
import torchmetrics

class Callback:
    """
    Base callback class that implements all supported hooks called during training
    
    All custom callbacks should inherit from this class. 
    """
    
    def __init__(self) -> None:
        pass
    
    def on_fit_start(self, trainer, model):
        pass
    
    def on_fit_end(self, trainer, model):
        pass
    
    def on_fit_epoch_end(self, trainer, model):
        pass
    
    
    def on_train_epoch_start(self, trainer, model):
        pass
    
    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        pass
    
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        pass
    
    def on_train_epoch_end(self, trainer, model):
        pass
    
    
    def on_validation_epoch_start(self, trainer, model):
        pass
    
    def on_validation_batch_start(self, trainer, model, batch, batch_idx):
        pass
    
    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self, trainer, model):
        pass
    
    
    def on_predict_start(self, trainer, model):
        pass
    
    def on_predict_epoch_start(self, trainer, model):
        pass
    
    def on_predict_batch_start(self, trainer, model, batch, batch_idx):
        pass
    
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx):
        pass
    
    def on_predict_epoch_end(self, trainer, model):
        pass

    def on_predict_end(self, trainer, model):
        pass
    

class BaseEvaluator(Callback):
    """ 
    Base evaluator class. Implements basic functionality for storing the outputs, predictions and inputs
    (if enabled) on each stage of the training and predict loops. It also logs automatically the computed
    metrics on each step/epoch to the trainer
    
    It is recommended to inherit from this class and implement the calc_step_metrics and calc_epoch_metrics.
    """
    
    def __init__(self, store_training_inputs=False, store_validation_inputs=False, store_predict_inputs=True) -> None:
        super().__init__()
        self.store_training_inputs = store_training_inputs
        self.store_validation_inputs = store_validation_inputs
        self.store_predict_inputs = store_predict_inputs
        
    def calc_step_metrics(self, trainer, model, outputs, labels, stage, batch_idx) -> list[tuple]:
        raise NotImplementedError
    
    def calc_epoch_metrics(self, trainer, model, outputs, labels, stage) -> list[tuple]:
        raise NotImplementedError
    
    def __register_variable(self, name:str, variable):
        previous_val = getattr(self, name)
        if isinstance(variable, torch.Tensor):
            if previous_val is None:
                setattr(self, name, variable)
            else:
                concated = self.concat_tensor_variable(name, previous_val, variable)
                setattr(self, name, concated)
        else:
            if previous_val is None:
                setattr(self, name, [variable])
            else:
                setattr(self, name, previous_val.append(variable))
                
    def concat_tensor_variable(self, name, previous_value, next_value):
        return torch.cat([previous_value, next_value], dim=0)
        
    def on_train_epoch_start(self, trainer, model):
        self.epoch_training_inputs = None
        self.epoch_training_outputs = None
        self.epoch_training_labels = None
        
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.store_training_inputs:
            self.__register_variable('epoch_training_inputs', batch[0])
        self.__register_variable('epoch_training_outputs', outputs)
        self.__register_variable('epoch_training_labels', batch[1])
        
        metrics = self.calc_step_metrics(trainer, model, outputs, batch[1], 'train', batch_idx)
        trainer.log(metrics)
        
    def on_train_epoch_end(self, trainer, model):
        metrics = self.calc_epoch_metrics(trainer, model, self.epoch_training_outputs, self.epoch_training_labels, 'train')
        trainer.log(metrics)
    
    def on_validation_epoch_start(self, trainer, model):
        self.epoch_validation_inputs = None
        self.epoch_validation_outputs = None
        self.epoch_validation_labels = None
        
    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.store_training_inputs:
            self.__register_variable('epoch_validation_inputs', batch[0])
        self.__register_variable('epoch_validation_outputs', outputs)
        self.__register_variable('epoch_validation_labels', batch[1])

        metrics = self.calc_step_metrics(trainer, model, outputs, batch[1], 'val', batch_idx)
        metrics = [('val_'+m, v) for m, v in metrics]
        trainer.log(metrics)
        
    def on_validation_epoch_end(self, trainer, model):
        metrics = self.calc_epoch_metrics(trainer, model, self.epoch_validation_outputs, self.epoch_validation_labels, 'val')
        metrics = [('val_'+m, v) for m, v in metrics]
        trainer.log(metrics)
        
    def on_predict_epoch_start(self, trainer, model):
        self.epoch_predict_inputs = None
        self.epoch_predict_outputs = None
        self.epoch_predict_labels = None
        
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.store_predict_inputs:
            self.__register_variable('epoch_predict_inputs', batch[0])
        self.__register_variable('epoch_predict_outputs', outputs)
        self.__register_variable('epoch_predict_labels', batch[1])
        
        metrics = self.calc_step_metrics(trainer, model, outputs, batch[1], 'predict', batch_idx)
        trainer.log(metrics)
        
    def on_predict_epoch_end(self, trainer, model):
        metrics = self.calc_epoch_metrics(trainer, model, self.epoch_predict_outputs, self.epoch_predict_labels, 'predict')
        trainer.log(metrics)
    
    
class TorchMetricsEvaluator(BaseEvaluator):
    """ 
    Class for evaluating the metrics of a model using torchmetrics and log the results to 
    the trainer.
    
    Use the add_step_metrics and add_epoch_metric methods to add the metrics to evaluate.
    """
    
    def __init__(self, store_training_inputs=False, store_validation_inputs=False, store_predict_inputs=True) -> None:
        super().__init__(store_training_inputs, store_validation_inputs, store_predict_inputs)
        self.step_metrics:torchmetrics.MetricCollection = None
        self.epoch_metrics:torchmetrics.MetricCollection = None
        
    def add_step_metrics(self, metrics:dict[str, torchmetrics.Metric]):
        if self.step_metrics is None:
            self.step_metrics = torchmetrics.MetricCollection(metrics)
        else:
            self.step_metrics.add_metrics(metrics)
    
    def add_epoch_metrics(self, metrics:dict[str, torchmetrics.Metric]):
        if self.epoch_metrics is None:
            self.epoch_metrics = torchmetrics.MetricCollection(metrics)
        else:
            self.epoch_metrics.add_metrics(metrics)
    
    def calc_step_metrics(self, trainer, model, outputs, labels, stage, batch_idx) -> list[tuple]:
        if self.step_metrics is None: return []
        metric_collection = self.step_metrics.to(outputs.device)
        metrics = metric_collection(outputs, labels)
        return [(m, v.item()) for m,v in metrics.items()]
    
    def calc_epoch_metrics(self, trainer, model, outputs, labels, stage) -> list[tuple]:
        if self.epoch_metrics is None: return []
        metric_collection = self.epoch_metrics.to(outputs.device)
        metrics = metric_collection(outputs, labels)
        return [(m, v.item()) for m,v in metrics.items()]
            