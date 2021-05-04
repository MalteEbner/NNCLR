# NNCLR


## Basic implementation:
`source/nn_memory_bank.py` implements a memory bank with nearest neigbours. In its forward pass, you input an output which updates the bank. Then the nearest neighbours to the output are returned.

`source/ModuleWithNN.py` should replace the `nn.Modules`we usually use by wrapping around them. Its `forward()` function calls are passed to the `base_module`, and then the output of it is replaced by the one nearest neighbours.
Big problem here: There are a huge number of different return formats of the forward function and thus wrapping around it is very difficult.
Perhaps it would be easier to wrap around a loss and its `forward()` function, as this function has a very clear input format: two `torch.Tensors` 

`nn_with_simsiam.py` is mostly the same file as from https://github.com/IgorSusmelj/simsiam-cifar10/blob/main/main.py. It is changed to replace the default `SimSiamModel.resnet_simsiam` with the `ModuleWithNN` wrapper around it. 
