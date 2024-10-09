from easydict import EasyDict


class SerializableEasyDict(EasyDict):
    def state_dict(self):
        return dict(self)

    def load_state_dict(self, state_dict):
        self.update(state_dict)
