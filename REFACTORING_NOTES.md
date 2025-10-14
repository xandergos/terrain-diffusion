# Training System Refactoring

This document describes the major refactoring of the training system that consolidates all training scripts into a unified framework.

## Overview

The training system has been refactored to use a unified `train.py` script with trainer classes for different model types. This improves code organization, reduces duplication, and makes the system easier to maintain and extend.

**Key improvements:**
- ✅ **Zero conditional logic in train.py** - No `if trainer_type == 'xyz'` checks
- ✅ **EMA managed by trainers** - Each trainer handles its own EMA initialization and lifecycle
- ✅ **Clean interface** - Trainers implement a simple, well-defined interface
- ✅ **Self-contained trainers** - Each trainer is fully responsible for its own setup and teardown

## Changes

### 1. Unified Training Script

All training is now handled through a single entry point:
```bash
python -m terrain_diffusion.training.train -c <config_file>
```

The old training scripts are preserved but the new system should be used going forward:
- `terrain_diffusion/training/gan/train_gan.py` (old)
- `terrain_diffusion/training/autoencoder/train_ae.py` (old)
- `terrain_diffusion/training/diffusion/train.py` (old)
- `terrain_diffusion/training/consistency/distill.py` (old)

### 2. Trainer Architecture

All trainers inherit from a base `Trainer` class defined in `trainer.py`:

```python
class Trainer(abc.ABC):
    @abc.abstractmethod
    def train_step(self, state):
        """Perform one training step."""
        pass

    @abc.abstractmethod
    def evaluate(self):
        """Perform a full evaluation cycle, returning a dictionary of metrics."""
        pass
    
    @abc.abstractmethod
    def get_accelerate_modules(self):
        """Returns a tuple of modules that need to be passed to accelerator.prepare."""
        pass
    
    @abc.abstractmethod
    def set_prepared_modules(self, prepared_modules):
        """Set the modules returned from accelerator.prepare back into the trainer."""
        pass
    
    @abc.abstractmethod
    def get_checkpoint_modules(self):
        """Returns a list of modules that need to be registered for checkpointing."""
        pass
    
    @abc.abstractmethod
    def load_model_checkpoint(self, model_ckpt_path):
        """Load model weights from a checkpoint file."""
        pass
    
    @abc.abstractmethod
    def get_model_for_saving(self):
        """Returns the main model to save config for."""
        pass
```

Implemented trainers:
- `GANTrainer` - for GAN models
- `AutoencoderTrainer` - for autoencoder models
- `DiffusionTrainer` - for diffusion models (both base and decoder)
- `ConsistencyTrainer` - for consistency distillation models

### 3. Code Organization

**Shared Code (in `train.py`):**
- Config loading and CLI argument parsing
- W&B initialization and logging
- Checkpoint saving/loading
- Accelerator setup
- Main training loop
- Progress bars and metrics aggregation

**Trainer-Specific Code (in trainer classes):**
- Model initialization
- Dataset setup
- Optimizer creation
- EMA initialization and management
- Loss calculation
- Training step implementation
- Evaluation logic
- Model-specific metrics
- Module preparation (unpacking accelerator.prepare results)
- Checkpoint module registration

### 4. Config Updates

All config files have been updated to specify which trainer to use:

**For .cfg files:**
```ini
[trainer_class]
@trainer=gan  # or autoencoder, diffusion, consistency
```

**For .yaml files:**
```yaml
trainer_class:
  "@trainer": "gan"  # or autoencoder, diffusion, consistency
```

### 5. Registry Updates

The registry now includes trainer registration:
```python
registry.trainer = catalogue.create("confection", "trainers", entry_points=False)
registry.trainer.register("gan", func=GANTrainer)
registry.trainer.register("autoencoder", func=AutoencoderTrainer)
registry.trainer.register("diffusion", func=DiffusionTrainer)
registry.trainer.register("consistency", func=ConsistencyTrainer)
```

## Backwards Compatibility

### Checkpoint Compatibility

✅ **Old checkpoints are fully compatible with the new system.**

The refactored system maintains complete backwards compatibility with existing checkpoints because:

1. **Same checkpoint format**: We continue to use `accelerator.save_state()` and `accelerator.load_state()`, which saves:
   - Model state dict
   - Optimizer state dict(s)
   - EMA state
   - Training state (epoch, step, seen)
   
2. **Same file structure**: Checkpoints still contain:
   - `pytorch_model.bin` or equivalent for model weights
   - `optimizer.bin` for optimizer state
   - `phema.pt` for EMA state
   - `config.json` for training config
   - `model_config/` for model architecture
   - `wandb_run.json` for W&B run ID

3. **Automatic resumption**: The system automatically detects existing checkpoints and prompts for resumption.

### Config Compatibility

⚠️ **Old configs need minor updates to work with the new system.**

To use the new unified training script with existing configs, add a `trainer_class` field at the top:

```ini
# For .cfg files
[trainer_class]
@trainer=<trainer_type>
```

```yaml
# For .yaml files
trainer_class:
  "@trainer": "<trainer_type>"
```

Where `<trainer_type>` is one of: `gan`, `autoencoder`, `diffusion`, `consistency`

All configs in the repository have been updated accordingly.

### Using Old Training Scripts

The old training scripts are still present and functional if needed. However, the new unified system is recommended for:
- Better code organization
- Easier maintenance
- More consistent behavior across model types
- Future extensibility

## Usage Examples

### Training a GAN
```bash
python -m terrain_diffusion.training.train -c configs/gan/gan.cfg
```

### Training an Autoencoder
```bash
python -m terrain_diffusion.training.train -c configs/autoencoder/autoencoder_x8.yaml
```

### Training a Diffusion Model
```bash
python -m terrain_diffusion.training.train -c configs/diffusion_base/diffusion_128-3.cfg
```

### Training a Consistency Model
```bash
python -m terrain_diffusion.training.train -c configs/consistency_base/consistency_base_128-3.cfg
```

### Resuming from Checkpoint
```bash
python -m terrain_diffusion.training.train -c <config> --ckpt <checkpoint_dir>
```

### With Config Overrides
```bash
python -m terrain_diffusion.training.train -c <config> -o training.epochs=100 -o training.batch_size=64
```

## Testing

To verify the refactoring works correctly:

1. **Test checkpoint loading**: Load an old checkpoint and verify training resumes correctly
2. **Test new training**: Start fresh training runs for each model type
3. **Test config overrides**: Verify CLI overrides work as expected
4. **Test evaluation**: Verify evaluation metrics are computed correctly

## Design Philosophy

The refactored system follows these key principles:

1. **No conditionals in shared code** - `train.py` has zero `if trainer_type == 'xyz'` checks. All trainer-specific logic is delegated to the trainer classes through the abstract interface.

2. **Self-contained trainers** - Each trainer is fully responsible for its own lifecycle:
   - Initializes its own models, datasets, optimizers, and EMA
   - Unpacks its own accelerator-prepared modules
   - Knows which modules need checkpointing
   - Handles its own model loading
   
3. **Clean interface** - The `Trainer` ABC defines a minimal but complete interface that all trainers must implement, making it easy to add new trainer types.

4. **Separation of concerns** - `train.py` handles orchestration (config, logging, checkpointing, training loop), while trainers handle implementation details (loss functions, model architectures, optimization strategies).

## Future Extensions

The new architecture makes it easy to add new trainers:

1. Create a new trainer class inheriting from `Trainer`
2. Implement all abstract methods:
   - `train_step()` - Training logic
   - `evaluate()` - Evaluation logic  
   - `get_accelerate_modules()` - What to prepare
   - `set_prepared_modules()` - How to unpack prepared modules
   - `get_checkpoint_modules()` - What to checkpoint (usually just EMA)
   - `load_model_checkpoint()` - How to load pretrained weights
   - `get_model_for_saving()` - Which model to save config for
3. Register it in `registry.py`
4. Add `trainer_class` to configs

Example trainers that could be added:
- `DiffusionDecoderTrainer` (if decoder-specific logic is needed)
- `ConsistencyDecoderTrainer` (if decoder-specific logic is needed)
- `CLIPGuidedTrainer` (for CLIP-guided generation)
- `MultiStageTrainer` (for multi-stage training pipelines)
- `LoRATrainer` (for LoRA fine-tuning)

