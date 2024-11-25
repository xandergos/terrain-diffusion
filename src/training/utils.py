from easydict import EasyDict


class SerializableEasyDict(EasyDict):
    def state_dict(self):
        return {k: v for k, v in self.items() if k not in ['state_dict', 'load_state_dict']}

    def load_state_dict(self, state_dict):
        self.update(state_dict)
