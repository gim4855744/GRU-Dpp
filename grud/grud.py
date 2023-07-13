from .base import GRUDBase

__all__ = ['GRUD', 'GRUDpp']


class GRUD(GRUDBase):
    def __init__(self, in_size, hidden_size, lr, weight_decay):
        super().__init__(in_size, hidden_size, lr, weight_decay, pp_mode=False)


class GRUDpp(GRUDBase):
    def __init__(self, in_size, hidden_size, lr, weight_decay):
        super().__init__(in_size, hidden_size, lr, weight_decay, pp_mode=True)
