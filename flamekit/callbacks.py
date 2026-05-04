
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
        
    def calc_metrics(self, trainer, model, outputs, batch, batch_idx, stage) -> list[tuple]:
        raise NotImplementedError
    
    def reset_metrics(self, trainer, model, stage:str):
        pass
        
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch, batch_idx, trainer.TRAIN)
        trainer.log(metrics)
    
    def on_train_epoch_end(self, trainer, model):
        self.reset_metrics(trainer, model, trainer.TRAIN)
        
    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch, batch_idx, trainer.VAL)
        metrics = [('val_'+m, v) for m, v in metrics]
        trainer.log(metrics)
        
    def on_validation_epoch_end(self, trainer, model):
        self.reset_metrics(trainer, model, trainer.VAL)
        
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metrics = self.calc_metrics(trainer, model, outputs, batch, batch_idx, trainer.PREDICT)
        trainer.log(metrics)
        
    def on_predict_epoch_end(self, trainer, model):
        self.reset_metrics(trainer, model, trainer.PREDICT)


class TorchMetricsEvaluator(BaseEvaluator):
    """
    Evaluator that computes metrics using torchmetrics and logs them to the trainer.

    Supports two tiers of metrics:
        - Step metrics: computed and logged every batch (e.g. MAE, MSE).
        - Epoch metrics: accumulated every batch via update() and computed once
          at the end of each epoch (e.g. R2Score).

    Use add_metrics() for step-level metrics and add_epoch_metrics() for
    epoch-level metrics.
    """

    def __init__(self) -> None:
        try:
            import torchmetrics
        except ImportError:
            raise ImportError('"torchmetrics" is not installed')
        super().__init__()
        self.step_metrics: torchmetrics.MetricCollection = None
        self.epoch_metrics: torchmetrics.MetricCollection = None

    def add_metrics(self, metrics: dict):
        if self.step_metrics is None:
            self.step_metrics = torchmetrics.MetricCollection(metrics)
        else:
            self.step_metrics.add_metrics(metrics)

    def add_epoch_metrics(self, metrics: dict):
        if self.epoch_metrics is None:
            self.epoch_metrics = torchmetrics.MetricCollection(metrics)
        else:
            self.epoch_metrics.add_metrics(metrics)

    def calc_metrics(self, trainer, model, outputs, batch, batch_idx, stage):
        labels = batch[1]
        results = []

        if self.step_metrics is not None:
            collection = self.step_metrics.to(outputs.device)
            metrics = collection(outputs, labels)
            results.extend([(m, v.item()) for m, v in metrics.items()])

        if self.epoch_metrics is not None:
            self.epoch_metrics.to(outputs.device).update(outputs, labels)

        return results

    def _compute_epoch_metrics(self, trainer, stage):
        if self.epoch_metrics is None:
            return
        metrics = self.epoch_metrics.compute()
        prefix = "val_" if stage == trainer.VAL else ""
        trainer.log([(prefix + m, v.item()) for m, v in metrics.items()], average=False)

    def reset_metrics(self, trainer, model, stage: str):
        if self.step_metrics is not None:
            self.step_metrics.reset()
        if self.epoch_metrics is not None:
            self.epoch_metrics.reset()

    def on_train_epoch_end(self, trainer, model):
        self._compute_epoch_metrics(trainer, trainer.TRAIN)
        self.reset_metrics(trainer, model, trainer.TRAIN)

    def on_validation_epoch_end(self, trainer, model):
        self._compute_epoch_metrics(trainer, trainer.VAL)
        self.reset_metrics(trainer, model, trainer.VAL)

    def on_predict_epoch_end(self, trainer, model):
        self._compute_epoch_metrics(trainer, trainer.PREDICT)
        self.reset_metrics(trainer, model, trainer.PREDICT)