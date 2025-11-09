import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ema_pytorch import PostHocEMA
from tqdm import tqdm

from terrain_diffusion.training.trainers.trainer import Trainer
from terrain_diffusion.training.datasets import LongDataset
from terrain_diffusion.training.utils import temporary_ema_to_model, recursive_to


class PerceptronTrainer(Trainer):
    """Trainer for simple prediction with selectable loss.

    - For 'cce' (categorical cross entropy): targets are LongTensor shape (B,) with class indices,
      and model outputs logits of shape (B, num_classes).
    - For 'mse' or 'mae': targets are FloatTensor of shape (B, C), and model outputs predictions of shape (B, C).
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
        self.loss_type = config['training'].get('loss', 'cce').lower()
        if self.loss_type == 'cce':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_type == 'mae':
            self.criterion = torch.nn.L1Loss()
        elif self.loss_type == 'high_mae':
            def high_mae(x, y):
                high_score = 0.1 * torch.abs(x - y) * (torch.abs(x - y) > 0.0).float()
                low_score = 0.9 * torch.abs(x - y) * (torch.abs(x - y) <= 0.0).float()
                return (high_score + low_score).float().mean()
            self.criterion = high_mae
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Expected one of ['cce', 'mse', 'mae'].")

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
        """One optimization step using the configured loss.

        Expects batch to contain:
        - 'x': FloatTensor of shape (B, C)
        - 'y':
            * LongTensor of shape (B,) for 'cce'
            * FloatTensor of shape (B, C) for 'mse'/'mae'
        """
        x = batch['x']
        y = batch['y']

        self.model.train()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(x)
                if self.loss_type == 'cce':
                    loss = self.criterion(outputs, y.long())
                else:
                    loss = self.criterion(outputs.float(), y.float())

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

        metrics = {}
        with torch.no_grad():
            if self.loss_type == 'cce':
                preds = outputs.argmax(dim=1)
                acc = (preds == y.long()).float().mean().item()
                metrics['acc'] = acc
            else:
                # Report regression metrics for convenience
                mae_val = torch.nn.functional.l1_loss(outputs.float(), y.float(), reduction='mean').item()
                mse_val = torch.nn.functional.mse_loss(outputs.float(), y.float(), reduction='mean').item()
                metrics['mae'] = mae_val
                metrics['mse'] = mse_val

        return {
            'loss': loss.item(),
            **metrics,
            'lr': lr,
            'grad_norm': grad_norm.item(),
        }

    def evaluate(self):
        """Evaluate on validation set; report mean loss/metrics using EMA weights and eval() mode."""
        # Reuse training dataset for evaluation
        val_dataset = LongDataset(self.train_dataset, shuffle=True)
        val_dataset.set_seed(self.config['training'].get('seed', 0) + 638)
        batch_size = self.config['training']['batch_size']
        dataloader_kwargs = dict(self.resolved.get('dataloader_kwargs', {}))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, **dataloader_kwargs)
        val_iter = iter(val_loader)
        
        # Number of samples to process
        eval_cfg = self.config.get('evaluation', {})
        total_samples = eval_cfg.get('validation_steps', batch_size * 1024)
        
        processed = 0
        losses = []
        correct = 0
        total = 0
        mae_vals, mse_vals = [], []
        
        with temporary_ema_to_model(self.ema.ema_models[0]):
            self.model.eval()
            pbar = tqdm(total=total_samples, desc="Validation")
            while processed < total_samples:
                batch = recursive_to(next(val_iter), device=self.accelerator.device)
                x, y = batch['x'], batch['y']
                
                with torch.no_grad(), self.accelerator.autocast():
                    outputs = self.model(x)
                    if self.loss_type == 'cce':
                        loss = self.criterion(outputs, y.long())
                    else:
                        loss = self.criterion(outputs.float(), y.float())
                
                losses.append(loss.item())
                if self.loss_type == 'cce':
                    preds = outputs.argmax(dim=1)
                    correct += (preds == y.long()).sum().item()
                    total += y.shape[0]
                else:
                    mae_vals.append(torch.nn.functional.l1_loss(outputs.float(), y.float(), reduction='mean').item())
                    mse_vals.append(torch.nn.functional.mse_loss(outputs.float(), y.float(), reduction='mean').item())
                
                inc = min(x.shape[0], total_samples - processed)
                processed += inc
                pbar.update(inc)
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
            pbar.close()
        
        metrics = {'val/loss': float(np.mean(losses)) if len(losses) > 0 else 0.0}
        if self.loss_type == 'cce':
            if total > 0:
                metrics['val/acc'] = correct / total
        else:
            if len(mae_vals) > 0:
                metrics['val/mae'] = float(np.mean(mae_vals))
            if len(mse_vals) > 0:
                metrics['val/mse'] = float(np.mean(mse_vals))
        
        # Print final results
        if self.loss_type == 'cce':
            acc_str = f", acc: {metrics.get('val/acc', 0.0):.4f}" if 'val/acc' in metrics else ""
            print(f"Validation - loss: {metrics['val/loss']:.4f}{acc_str}")
        else:
            mae_str = f", mae: {metrics.get('val/mae', 0.0):.4f}" if 'val/mae' in metrics else ""
            mse_str = f", mse: {metrics.get('val/mse', 0.0):.4f}" if 'val/mse' in metrics else ""
            print(f"Validation - loss: {metrics['val/loss']:.4f}{mae_str}{mse_str}")
        
        torch.cuda.empty_cache()
        return metrics