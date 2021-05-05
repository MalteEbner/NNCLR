# NNCLR


## Basic implementation:
`source/nn_memory_bank.py` implements a memory bank with nearest neigbours. In its forward pass, you input an output which updates the bank. Then the nearest neighbours to the output are returned.

`nn_with_simsiam.py` is mostly the same file as from https://github.com/IgorSusmelj/simsiam-cifar10/blob/main/main.py. It is changed to replace the embeddings for out0 with their nearest neighbours from the memory bank.
