import torch.nn as nn

# PyTorch automatically initializes weights and biases, this func is provided in case other initialization is needed.
# An example of a linear layer using Xavier (Glorot) uniform initialization
#model = FCN([784, 50, 20, 10])
#model.apply(initialize_weights)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

def print_model_weights(model):
    params_list = list(model.parameters())
    print(f"Total number of parameter groups: {len(params_list)}")

    for i, param in enumerate(params_list):
        print(f"{i} - Parameter shape: {param.size()}")

        print(f"Parameter values:\n{param}")