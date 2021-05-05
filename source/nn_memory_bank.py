import torch
from lightly.loss.memory_bank import MemoryBankModule


class NNmemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16):
        super(NNmemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False,
                normalize: bool = True):

        if normalize:
            output = torch.nn.functional.normalize(output, dim=1)

        output, bank = super(NNmemoryBankModule, self).forward(output, labels, update)
        bank = bank.to(output.device).t()

        similarity_matrix = torch.einsum("nd,md->nm", output, bank)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours
