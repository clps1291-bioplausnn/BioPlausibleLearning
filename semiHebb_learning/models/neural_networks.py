import os
import torch
import torch.nn as nn

from models.learning_rules import *

# ---------------------------------------------------------- Stage 1: Basic Components --------------------------------------------------------------
     
# A naive fully connected nn (FCN/MLP) without activation functions
class FCN(nn.Module):
    def __init__(self, layer_sizes , activation = None, pretrained_model_path=None):
        super(FCN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if activation and i < len(layer_sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)

    def forward(self, x): 
        return self.model(x) 


# A naive HebbNet: Pure HebbLayers
class HebbNet(nn.Module):
    def __init__(self, layer_sizes, lr, require_hebb=True, activation=True, update_rule='hebb', p=None, pretrained_model_path=None):
        super(HebbNet, self).__init__()
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(HebbLayer(layer_sizes[i], layer_sizes[i+1], lr=lr, require_hebb=require_hebb, activation=activation, update_rule=update_rule, p=p))
        self.model = nn.Sequential(*self.layers)
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
    
    # A helper function to ensure that the change of require_hebb property is propagated to all the constituent HebbLayers
    def set_require_hebb(self, require_hebb):
        for layer in self.layers:
            if isinstance(layer, HebbLayer):
                layer.require_hebb = require_hebb

# A naive ConvLayers: there is no fully connected layers here.
class ConvLayers(nn.Module):
    def __init__(self, conv, pretrained_model_path=None):
        super(ConvLayers, self).__init__()
        layers = []
        for params in conv:
            in_channels, out_channels, kernel_size, stride, padding = params
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size= 2) # let's keep it simple i.e. using the same maxpooling kernel
            ))
        self.model = nn.Sequential(*layers)
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)


    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------- Stage 2: Put things together --------------------------------------------------------------
# A naive CNN: Conv2d, ReLU, MaxPool2d, fc layers
class CNN(nn.Module):
    def __init__(self, ConvLayers, FCN, pretrained_model_path=None):
        super(CNN, self).__init__()
        self.conv_layers = ConvLayers
        self.fc = FCN

        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv_layers(x) 
        x = x.view(x.size(0), -1)           # flatten
        output = self.fc(x)  
 
        return output,x
    
# A naive SemiHebbNet: a hybrid approach composed of Hebbian learning + bp trained fc layers 
class SemiHebbNet(nn.Module):
    def __init__(self, HebbNet, FCN, pretrained_model_path=None):
        super(SemiHebbNet, self).__init__()
        self.HebbNet = HebbNet
        self.FCN = FCN
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)


    def forward(self, x):
        x = self.HebbNet(x)
        output = self.FCN(x)
        return output,x
    
    # Propagate the change of require_hebb property to the HebbNet component
    def set_require_hebb(self, require_hebb):
        if hasattr(self.HebbNet, 'set_require_hebb'):
            self.HebbNet.set_require_hebb(require_hebb)
    



# Krotov

# ConvLayers + FA
# ---------------------------------------------------------- Stage 3: Wild Explorations --------------------------------------------------------------