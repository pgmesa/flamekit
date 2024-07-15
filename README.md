# FlameKit

[![Downloads](https://static.pepy.tech/personalized-badge/flamekit?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/flamekit)

FlameKit is a minimalistic toolkit for PyTorch, created to streamline the training and evaluation process. It is designed to eliminate the boilerplate code needed for looping over datasets, logging metrics, and plotting results. Each critical part of the training and evaluation phases is implemented in a different compartmentalized function, which can be overridden to cater to specific use cases. It is intended to be lightweight, fast, and highly customizable.

FlameKit provides a trainer class, callbacks with predefined hooks, functionality for setting up a reproducible environment, customizable progress bars, learning rate schedulers, and more. Its API is similar to PyTorch Lightning's, but it prioritizes minimal code and lightweight design.

Check the `/examples` directory for more detailed information on how to use this package.

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using Evaluator Callback](#using-evaluator-callback)
- [Metrics Logging and Plots](#metrics-logging-and-plots)
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
from flamekit.trainer import TorchTrainer
from flamekit.callbacks import Callback
from flamekit.pbars import TQDMProgressBar
from flamekit.utils import get_next_experiment_path, set_up_reproducible_env
from flamekit.lr_scheduler import LRScheduler, CosineDecay, LinearDecay

set_up_reproducible_env(seed=1337)

total_it = epochs * len(train_loader)
warmup_it = warmup_epochs * len(train_loader)      # Warmup iterations
cooldown_it = cooldown_epochs * len(train_loader)  # Cooldown iterations
lr_decay_it = total_it - cooldown_it
lr_decay_fn = CosineDecay(k=2)
    
class TrainingStrategy(Callback):
    
    def __init__(self) -> None:
        self.lr_scheduler = LRScheduler(
            lr0, lrf, lr_decay_it, warmup_it=warmup_it, decay_fn=lr_decay_fn
        )
        
    def on_fit_start(self, trainer, model):
        self.lr_scheduler.reset()
    
    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        # Update lr
        new_lr = self.lr_scheduler.step()
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = new_lr
        # Monitor lr
        trainer.log([('lr', new_lr)], average=False)

trainer = TorchTrainer(model, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr0)
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
Epoch 1/10: 100% |██████████████████████████████| 58/58 [00:30, lr=0.001, loss=1.96, val_loss=1.77]
[INFO] Saving best checkpoint, regarding 'val_loss' metric -- mode='min' (checkpoints\experiment_2\ckp_val-loss_1.7744_1_best.pt)
Epoch 2/10: 100% |██████████████████████████████| 58/58 [00:29, lr=0.001, loss=1.72, val_loss=1.79]
Epoch 3/10:  72% |█████████████████████▋        | 42/58 [00:22, lr=0.000999, loss=1.68]
...
```

## Using Evaluator Callback

Evaluator callbacks can be used to evaluate the model at each step or epoch and log the results to the trainer. You can create your own evaluators by inheriting from the `BaseEvaluator` class and implementing the `calc_metrics` and `reset_metrics` methods. Additionally, an in-built evaluator called `TorchMetricsEvaluator` is available, which accepts `torchmetrics` metrics. Here's how to use it:

```python
import torchmetrics
from flamekit.callbacks import TorchMetricsEvaluator
    
evaluator = TorchMetricsEvaluator()

class Accuracy(torchmetrics.Accuracy):
    def update(self, preds, target):
        preds = preds.argmax(dim=1)
        super().update(preds, target)

metrics = {
    'acc': Accuracy(task=task, num_classes=n_classes, average=average),
    'precision': torchmetrics.Precision(task=task, num_classes=n_classes, average=average),
    'recall': torchmetrics.Recall(task=task, num_classes=n_classes, average=average),
    'f1': torchmetrics.F1Score(task=task, num_classes=n_classes, average=average),
    'auc': torchmetrics.AUROC(task=task, num_classes=n_classes, average=average),
}
evaluator.add_metrics(metrics)

callbacks = [evaluator, pbar]

history = trainer.fit(
    ...,
    callbacks=callbacks
)
```
```
Epoch 1/10: 100% |██████████████████████████████| 50/50 [00:56, loss=0.886, acc=0.746, auc=0.946, f1=0.726, precision=0.751, recall=0.746]
Epoch 2/10:  72% |█████████████████████▌        | 36/50 [00:35, loss=0.253, acc=0.923, auc=0.996, f1=0.92, precision=0.925, recall=0.923] 
```

## Metrics Logging and Plots
While training, all metrics are logged to a .csv file in the experiments directory. Right before finishing the training, all metrics are plotted and saved. You can easily plot the generated figure by calling:

```python
trainer.plot()
```

![Results Example](assets\results_example.png)

You can customize your own figures with different colors, select which metrics to show, and save them to a different file:

```python
trainer.plot(metrics=['f1', 'auc'], colors=[['#000000', '#1f77b4'], ['#2B2F42', '#EF233C']], dest_path=exp_path/'customization_example.png')
```

![Customization Example](assets\customization_example.png)


You can also compare the results of different experiments with a few lines of code (`/examples/compare_results.ipynb`):

![Results Comparison Example](assets\compare_results.png)

## Extending Trainer Functionality

You can override the main trainer function to customize its behavior. For example, to create an Automatic Mixed Precision (AMP) Trainer:

```python
from flamekit.training import TorchTrainer

class AMPTrainer(TorchTrainer):
    
    def __init__(self, model, device, amp_dtype=torch.float16, scale=True) -> None:
        super().__init__(model, device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.amp_dtype = amp_dtype
        self.scale = scale
    
    def training_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        with torch.autocast(device_type=inputs.device.type, dtype=self.amp_dtype):
            outputs = self.model(inputs)
            step_loss = self.loss_step(outputs, labels)
        return outputs, step_loss

    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        if self.scale:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
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
Epoch 1/10: 100% |██████████████████████████████| 50/50 [00:00<00:00, 91.17 steps/s, loss=0.278]
Epoch 2/10: 100% |██████████████████████████████| 50/50 [00:00<00:00, 72.83 steps/s, loss=0.166] 
Epoch 3/10: 100% |██████████████████████████████| 50/50 [00:00<00:00, 96.45 steps/s, loss=0.0967]
```
It also implements a `KerasProgressBar` class, which inherits from `TQDMProgressBar` and tries to replicate the Keras design:
```python
from flamekit.pbars import KerasProgressBar 

# Customize the progress bar
pbar = KerasProgressBar(pbar_size:int=30, ascii='.>=', desc_above=True, show_desc=True,
                 show_elapsed_time=True, show_rate=True, show_postfix=True, show_n_fmt=True,
                 show_total_fmt=True, pbar_frames=('[', ']'))

history = trainer.fit(
    ...,
    callbacks=[pbar]
)
```
```
Epoch 1/10
50/50 [==============================] - 00:00 77.64 steps/s, loss=0.303 
Epoch 2/10
50/50 [==============================] - 00:00 95.95 steps/s, loss=0.172
Epoch 3/10
50/50 [==============================] - 00:00 90.37 steps/s, loss=0.104 
```

Additionally, you can inspect `pbars.py` file to see how to create your own Progress Bar designs.


