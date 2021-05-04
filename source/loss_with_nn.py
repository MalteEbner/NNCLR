import torch
from torch import nn

from source.nn_memory_bank import NNmemoryBankModule


class LossWithNN(torch.nn.Module):
    def __init__(self, base_loss: nn.Module, nn_memory_bank_size: int = 2 ** 8):
        super(LossWithNN, self).__init__()
        self.base_loss = base_loss
        self.nn_memory_bank = NNmemoryBankModule(size=nn_memory_bank_size)

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):

        if isinstance(out0, tuple):
            z0, p0 = out0
            z0 = torch.nn.functional.normalize(z0, dim=1)
            p0 = torch.nn.functional.normalize(p0, dim=1)
            z0, bank = self.nn_memory_bank.forward(z0, update=True)
            p0, bank = self.nn_memory_bank.forward(p0, update=True)
            out0 = (z0, p0)

        elif isinstance(out0, torch.Tensor):
            out0 = torch.nn.functional.normalize(out0, dim=1)
            out0, bank = self.nn_memory_bank.forward(out0, update=True)

        loss = self.base_loss.forward(out0, out1)
        return loss
