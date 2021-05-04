import torch
from torch import nn

from source.nn_memory_bank import NNmemoryBankModule


class ModuleWithNN(nn.Module):
    def __init__(self, base_module: nn.Module, nn_memory_bank_size: int = 2 ** 8):
        super(ModuleWithNN, self).__init__()

        self.base_module = base_module
        self.nn_memory_bank = NNmemoryBankModule(size=nn_memory_bank_size)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):

        # quickfix
        assert return_features is False

        # forward pass
        ret_value = self.base_module.forward(x0, x1, return_features)

        if isinstance(ret_value, tuple):
            out0, out1 = ret_value
        else:
            out0 = ret_value

        # replace with nearest neighbour
        out0, bank = self.nn_memory_bank.forward(out0, update=True)

        if isinstance(ret_value, tuple):
            return out0, out1
        else:
            return out0
