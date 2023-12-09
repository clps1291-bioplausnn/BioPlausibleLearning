import torch.nn as nn

# PyTorch automatically initializes weights and biases, this func is provided in case other initialization is needed.
# An example of a linear layer using Xavier (Glorot) uniform initialization
#model = FCN([784, 50, 20, 10])
#model.apply(initialize_weights)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)