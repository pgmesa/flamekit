# FlameKit

FlameKit is a minimalistic toolkit for PyTorch, created to streamline the training process. It provides a Trainer class, Callbacks with predefined hooks, functionality for setting up a reproducible environment, evaluator callbacks, and a customizable progress bar. Its API is similar to PyTorch Lightning's, but it prioritizes minimal code, lightweightness, and ease of customization.

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using Evaluator Callback](#using-evaluator-callback)
- [Extending Trainer Functionality](#extending-trainer-functionality)
- [Customizable Progress Bars](#customizable-progress-bars)


## Installation

You can install FlameKit via pip:

```bash
pip install flamekit
```

## Quick Start

Here's a simple example demonstrating how to train a PyTorch model using FlameKit alongside custom callbacks. For more detailed examples explore the `/examples` directory:

```python
from flamekit.training import TorchTrainer
from flamekit.callbacks import Callback
from flamekit.utils import get_next_experiment_path, setup_reproducible_env

setup_reproducible_env(seed=1337)

# Define your custom callbacks
class TrainingStrategy(Callback):
    
    def __init__(self) -> None:
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, min_lr=min_lr)
    
    def update_lr(self, monitored_metric_value):
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step(monitored_metric_value)
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"[LR] Learning rate has changed from {current_lr} to {new_lr}")
    
    def on_train_epoch_start(self, trainer, model):
        dataset.enable_augment()

    def on_validation_epoch_start(self, trainer, model):
        dataset.disable_augment()
            
    def on_fit_epoch_end(self, trainer, model):
        monitored_metric_value = trainer.history[trainer.monitor][-1]
        self.update_lr(monitored_metric_value)
            
    def on_predict_epoch_start(self, trainer, model):
        dataset.disable_augment()

trainer = TorchTrainer(model, device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

trainer.compile(optimizer, criterion=criterion)

# Train your model
strategy = TrainingStrategy()
pbar = TQDMProgressBar(show_remaining_time=False, show_rate=False)
callbacks = [strategy, pbar]

history = trainer.fit(
    train_loader,
    epochs=epochs,
    validation_loader=val_loader,
    monitor='val_loss',
    dest_path=get_next_experiment_path('./checkpoints'),
    prefix=model.__class__.__name__,
    save_best=True,
    callbacks=callbacks
)
```
```
Epoch 1/10: 100% |██████████████████████████████| 58/58 [00:30, loss=1.96, val_loss=1.77]
[INFO] Saving best checkpoint, regarding 'val_loss' metric -- mode='min' (checkpoints\experiment_2\YoloV2_val-loss_1.7744_1_best.tar)
Epoch 2/10: 100% |██████████████████████████████| 58/58 [00:29, loss=1.72, val_loss=1.79]
Epoch 3/10:  72% |█████████████████████▋        | 42/58 [00:22, loss=1.68]
...
```

## Using Evaluator Callback

Evaluator callbacks can be used to evaluate the model at each step or epoch and log the results to the trainer. You can create your own evaluators by inheriting from the BaseEvaluator class and implementing the `calc_step_metrics` and `calc_epoch_metrics` methods. Additionally, an in-built evaluator called `TorchMetricsEvaluator` is available, which accepts torchmetrics metrics. Here's how to use it:

```python
import torchmetrics
from flamekit.callbacks import TorchMetricsEvaluator

evaluator = TorchMetricsEvaluator()

step_metrics = {
    'acc': torchmetrics.Accuracy(task=task, num_classes=n_classes, average=average),
    'precision': torchmetrics.Precision(task=task, num_classes=n_classes, average=average),
    'recall': torchmetrics.Recall(task=task, num_classes=n_classes, average=average),
}
epoch_metrics = {
    'f1': torchmetrics.F1Score(task=task, num_classes=n_classes, average=average),
    'auc': torchmetrics.AUROC(task=task, num_classes=n_classes, average=average),
}
evaluator.add_step_metrics(step_metrics)
evaluator.add_epoch_metric(epoch_metrics)

callbacks = [evaluator, pbar]

history = trainer.fit(
    ...,
    callbacks=callbacks
)
```
```
Epoch 5/10: 100% |██████████████████████████████| 50/50 [00:01, loss=0.278, acc=0.887, precision=0.878, recall=0.887, auc=0.954, f1=0.888]
Epoch 6/10:  92% |██████████████████████████    | 46/50 [00:01, loss=0.166, acc=0.934, precision=0.931, recall=0.931]
```

## Extending Trainer Functionality

You can override the main trainer function to customize its behavior. For example, to create an Automatic Mixed Precision Trainer:

```python
from flamekit.training import TorchTrainer

class AMPTrainer(TorchTrainer):
    
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
```

## Customizable Progress Bars

FlameKit provides a highly customizable progress bar based on TQDM. Here's an example:

```python
from flamekit.pbars import TQDMProgressBar 

# Customize the progress bar
pbar = TQDMProgressBar(pbar_size:int=30, ascii=None, desc_above=False,
                 show_desc=True, show_elapsed_time=True, show_remaining_time=True, show_rate=True,
                 show_postfix=True, show_n_fmt=True, show_total_fmt=True, show_percentage=True,
                 pbar_frames=('|','|'), l_bar=None, r_bar=None)

history = trainer.fit(
    ...,
    callbacks=[pbar]
)
```
```
Epoch 1/10: 100% |██████████████████████████████| 58/58 [00:27, loss=2.96, val_loss=1.77]
Epoch 1/10: 100% |██████████████████████████████| 58/58 [00:27, loss=2.96, val_loss=1.77]
```
It also comes with a KerasProgressBar replica, which inherits from this class and tries to replicate Keras design:
```
Epoch 1/10: 100% |██████████████████████████████| 58/58 [00:27, loss=2.96, val_loss=1.77]
```
