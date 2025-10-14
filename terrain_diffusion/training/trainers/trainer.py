import abc

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