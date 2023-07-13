import torch

from torch.nn import Module, Parameter
from torch.nn import Linear

from torch.nn.init import uniform_

from ..base import Base


class DiagLinear(Module):

    def __init__(self, in_size):
        super().__init__()
        self._weight = Parameter(torch.empty(in_size))
        self._bias = Parameter(torch.empty(in_size))
        self.init_parameters()

    def init_parameters(self):
        uniform_(self._weight, -1e-4, 1e-4)
        uniform_(self._bias, -1e-4, 1e-4)

    def forward(self, x):
        weight = torch.diagflat(self._weight)
        x = x @ weight + self._bias
        return x


class GRUDCellBase(Module):

    def __init__(self, in_size, hidden_size, pp_mode):

        super().__init__()

        self._pp_mode = pp_mode

        if self._pp_mode:
            gate_size = 3 * in_size + hidden_size
            self._input_decay = Linear(in_size, in_size)
        else:
            gate_size = 2 * in_size + hidden_size
            self._input_decay = DiagLinear(in_size)
        self._hidden_decay = Linear(in_size, hidden_size)

        self._gate1 = Linear(gate_size, hidden_size)
        self._gate2 = Linear(gate_size, hidden_size)

        self._out_layer = Linear(gate_size, hidden_size)

    def forward(self, x, x_mask, x_last, x_last_mask, x_interval, h):

        input_decay = self._input_decay(x_interval)
        input_decay = -torch.relu(input_decay)
        input_decay = torch.exp(input_decay)
        if self._pp_mode:
            input_decay = input_decay * x_last_mask

        imputation = input_decay * x_last + (1 - input_decay) * x
        x = x_mask * x + (1 - x_mask) * imputation

        hidden_decay = self._hidden_decay(x_interval)
        hidden_decay = -torch.relu(hidden_decay)
        hidden_decay = torch.exp(hidden_decay)

        h = hidden_decay * h

        if self._pp_mode:
            gate_x = torch.concat([x, h, x_mask, x_last_mask], dim=1)
        else:
            gate_x = torch.concat([x, h, x_mask], dim=1)

        gate1 = torch.sigmoid(self._gate1(gate_x))
        gate2 = torch.sigmoid(self._gate2(gate_x))

        if self._pp_mode:
            x = torch.concat([x, gate1 * h, x_mask, x_last_mask], dim=1)
        else:
            x = torch.concat([x, gate1 * h, x_mask], dim=1)

        new_h = torch.tanh(self._out_layer(x))
        new_h = (1 - gate2) * h + gate2 * new_h

        return new_h


class GRUDBase(Base):

    def __init__(self, in_size, hidden_size, lr, weight_decay, pp_mode):
        super().__init__(hidden_size, lr, weight_decay)
        self._hidden_size = hidden_size
        self._cell = GRUDCellBase(in_size, hidden_size, pp_mode)
        self._output_layer = Linear(hidden_size, 1)

    def forward(self, x, x_mask, x_last, x_last_mask, x_interval):

        device = self.device
        batch_size, num_time_steps, _ = x.shape
        outputs = []

        h = torch.zeros(batch_size, self._hidden_size, device=device)
        for t in range(num_time_steps):
            h = self._cell(x[:, t],
                           x_mask[:, t],
                           x_last[:, t],
                           x_last_mask[:, t],
                           x_interval[:, t],
                           h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)

        outputs = self._output_layer(outputs)
        outputs = torch.sigmoid(outputs)

        return outputs
