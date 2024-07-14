
try: import torchmetrics
except ImportError: pass

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
    Base evaluator class.

    This class serves as a foundation for evaluating models during training, validation, 
    and prediction stages. It automatically logs the computed metrics at the end of
    each batch to the trainer.

    To use this class, inherit from it and implement the `calc_metrics` method to define how 
    metrics should be calculated in each stage based on the model's outputs and the true labels.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def calc_metrics(self, trainer, model, outputs, labels, stage, batch_idx) -> list[tuple]:
        raise NotImplementedError
        
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch[1], 'train', batch_idx)
        trainer.log(metrics)
        
    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch[1], 'val', batch_idx)
        metrics = [('val_'+m, v) for m, v in metrics]
        trainer.log(metrics)
        
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch[1], 'predict', batch_idx)
        trainer.log(metrics)
    
    
class TorchMetricsEvaluator(BaseEvaluator):
    """ 
    Class for evaluating the metrics of a model using torchmetrics and log the results to 
    the trainer.
    
    Use the add_step_metrics and add_epoch_metric methods to add the metrics to evaluate.
    """
    def __init__(self) -> None:
        try:
            import torchmetrics
        except ImportError:
            raise ImportError('"torchmetrics" is not installed')
        super().__init__()
        self.step_metrics:torchmetrics.MetricCollection = None
        
    def add_metrics(self, metrics:dict[str, torchmetrics.Metric]):
        if self.step_metrics is None:
            self.step_metrics = torchmetrics.MetricCollection(metrics)
        else:
            self.step_metrics.add_metrics(metrics)
    
    def calc_metrics(self, trainer, model, outputs, labels, stage, batch_idx) -> list[tuple]:
        if self.step_metrics is None: return []
        metric_collection = self.step_metrics.to(outputs.device)
        metrics = metric_collection(outputs, labels)
        return [(m, v.item()) for m,v in metrics.items()]
            