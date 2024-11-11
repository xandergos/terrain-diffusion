from diffusers.training_utils import EMAModel

class PowerEMAModel(EMAModel):
    def __init__(self, parameters, gamma=None, width=None):
        assert gamma is not None or width is not None, "Either gamma or width must be provided"
        
        if width is not None:
            from scipy.optimize import fsolve
            def eq(g):
                return width - (g + 1)**0.5 * (1 / (g + 2)) * (1 / (g + 3)**0.5)
            self.gamma = fsolve(eq, 1)[0]
            self.width = width
        else:
            self.gamma = gamma
            self.width = (gamma + 1)**0.5 * (1 / (gamma + 2)) * (1 / (gamma + 3)**0.5)
            
        super().__init__(parameters)

    def get_decay(self, t: int) -> float:
        step = max(1, t - self.update_after_step)
        cur_decay_value = (1 - 1/step)**(self.gamma - 1)
        return cur_decay_value