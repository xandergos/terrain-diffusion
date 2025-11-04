import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ema_pytorch import PostHocEMA

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import temporary_ema_to_model


class PerceptronTrainer(Trainer):
    """Trainer for simple categorical climate prediction (classification).

    Assumes inputs are shaped (B, C) and targets are shaped (B,) with class indices.
    Uses CrossEntropyLoss on logits produced by the model.
    """

    def __init__(self, config, resolved, accelerator, state):
        self.config = config
        self.resolved = resolved
        self.accelerator = accelerator
        self.state = state

        # Model and datasets
        self.model = resolved['model']
        self.train_dataset = resolved['train_dataset']

        # Optimizer and loss
        self.optimizer = self._build_optimizer(self.model, config)
        self.criterion = torch.nn.CrossEntropyLoss()

        # EMA for checkpointing/eval parity with other trainers
        resolved['ema']['checkpoint_folder'] = os.path.join(resolved['logging']['save_dir'], 'phema')
        self.ema = PostHocEMA(self.model, **resolved['ema'])

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Training classifier with {n_params} parameters.")

    def _build_optimizer(self, model, config):
        opt_cfg = config['optimizer']
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable parameters found for optimizer.")
        if opt_cfg['type'] == 'adam':
            return torch.optim.Adam(params, **opt_cfg['kwargs'])
        elif opt_cfg['type'] == 'adamw':
            return torch.optim.AdamW(params, **opt_cfg['kwargs'])
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")

    def get_accelerate_modules(self):
        """Modules to pass to accelerator.prepare."""
        return (self.model, self.optimizer)

    def set_prepared_modules(self, prepared_modules):
        """Receive modules returned by accelerator.prepare and store them."""
        self.model, self.optimizer = prepared_modules
        self.ema = self.ema.to(self.accelerator.device)

    def get_checkpoint_modules(self):
        """Return modules for accelerator.register_for_checkpointing."""
        return [self.ema]

    def load_model_checkpoint(self, model_ckpt_path):
        """Load model weights from a checkpoint or HF-like directory."""
        try:
            if hasattr(type(self.model), 'from_pretrained'):
                temp_state = type(self.model).from_pretrained(model_ckpt_path).state_dict()
            else:
                loaded = torch.load(model_ckpt_path, map_location='cpu')
                temp_state = loaded.get('state_dict', loaded)

            current = self.model.state_dict()
            filtered = {k: v for k, v in temp_state.items() if k in current and current[k].shape == v.shape}
            missing = [k for k in current.keys() if k not in filtered]
            if len(missing) > 0:
                print(f"Warning: {len(missing)} parameters not found in checkpoint; leaving initialized.")
            self.model.load_state_dict(filtered, strict=False)
        except Exception as e:
            print(f"Failed to load model checkpoint: {e}")

    def get_model_for_saving(self):
        return self.model

    def train_step(self, state, batch):
        """One optimization step using CrossEntropy over logits.

        Expects batch to contain:
        - 'x': FloatTensor of shape (B, C)
        - 'y': LongTensor of shape (B,)
        """
        x = batch['x']
        y = batch['y']

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)

            # Update train counters and LR
            batch_size = x.shape[0]
            state['seen'] += batch_size
            state['step'] += 1

            lr = self.resolved['lr_sched'].get(state['seen'])
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            # Backprop
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['training'].get('gradient_clip_val', 10.0)
                )
            else:
                grad_norm = torch.tensor(0.0)
            self.optimizer.step()

        if self.accelerator.is_main_process:
            self.ema.update()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        return {
            'loss': loss.item(),
            'acc': acc,
            'lr': lr,
            'grad_norm': grad_norm.item(),
        }

    def evaluate(self):
        """Evaluate on validation set; report mean loss and accuracy using EMA weights."""
        pass