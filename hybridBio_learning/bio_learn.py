import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from fa_linear import *

def get_data(data_type):
    mat = scipy.io.loadmat('mnist_all.mat')
    X=torch.zeros((0, 28 * 28), dtype=torch.float)
    y=torch.zeros(0, dtype=torch.long)
    for i in range(10): 
        X_i = torch.from_numpy(mat[data_type + str(i)].astype(float)).float()
        X = torch.cat((X, X_i))
        y_i = torch.full(size=(len(X_i),), fill_value=i, dtype=torch.long)
        y = torch.cat((y, y_i))
    return X / 255.0, y

def draw_weights(weights, n_cols, n_rows, sz=28, text=None):
    weights = weights.reshape((-1, sz, sz))
    indexes = np.random.randint(0, len(weights), n_cols*n_rows)
    weights = weights[indexes]
    fig=plt.figure(figsize=(10, 6))    
    HM=np.zeros((sz*n_rows,sz*n_cols))
    for idx in range(n_cols * n_rows):
        x, y = idx % n_cols, idx // n_cols
        HM[y*sz:(y+1)*sz,x*sz:(x+1)*sz]=weights[idx]
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    if text is not None: plt.title(text)
    plt.axis('off')
    fig.canvas.draw()   

def get_unsupervised_weights(X, n_hidden, n_epochs, batch_size, 
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2):
    sample_sz = X.shape[1]
    weights = torch.rand((n_hidden, sample_sz), dtype=torch.float, device='cuda')
    for epoch in range(n_epochs):    
        eps = learning_rate * (1 - epoch / n_epochs)
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()            
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input=torch.mm(W, mini_batch)            
            
            y = torch.argsort(tot_input, dim=0)            
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength            
                    
            xx = torch.sum(yl * tot_input,1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)                            
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights            
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision: nc = precision            
            weights += eps*(ds/nc)
    return weights

def run_test(train_X, train_y, test_X, test_y, model, epochs, batch_size=64, lr=1e-3, verbose=0, loss=None):
    print("hello")
    start = time()
    train_ds = TensorDataset(train_X, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
    optimizer = Adam(model.parameters(), lr=lr)
    if loss is None: loss = F.nll_loss
    trainer = create_supervised_trainer(model, optimizer, loss, device='cuda')
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'nll': Loss(loss)}, device='cuda')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if verbose <= 1: return
        evaluator.run(train_dl)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if verbose == 0: return
        evaluator.run(test_dl)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.COMPLETED)
    def log_completed_validation_results(engine):
        evaluator.run(test_dl)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Final Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.2f} Took: {:.0f}s".format(engine.state.epoch, avg_accuracy, avg_nll, time() - start))

    trainer.run(train_dl, max_epochs=epochs) 
   

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(len(x), 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class BioCell(nn.Module):
    def __init__(self, Wᵤᵢ, β, out_features):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ
        self.β = β
        self.supervised = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ):
        Wᵤᵢvᵢ = F.linear(vᵢ, self.Wᵤᵢ, None)
        hᵤ = F.relu(Wᵤᵢvᵢ)
        Sₐᵤ = self.supervised(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤ)
        return cₐ

class BioCell2(nn.Module):
    def __init__(self, Wᵤᵢ, β=.1, out_features=10):
        super().__init__()
        # Wᵤᵢ is the unsupervised pretrained weight matrix of shape: (2000, 28*28)        
        self.Wᵤᵢ = Wᵤᵢ.transpose(0, 1)
        self.β = β
        self.supervised = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ):        
        Wᵤᵢvᵢ = torch.matmul(vᵢ, self.Wᵤᵢ)
        hᵤ = F.relu(Wᵤᵢvᵢ)
        Sₐᵤ = self.supervised(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤ)
        return cₐ

class BioCell3_withFA(nn.Module):
    # Wᵤᵢ is the unsupervised pretrained weight matrix of shape: (2000, 28*28)
    def __init__(self, Wᵤᵢ, n=4.5, β=.01, out_features=10):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ.transpose(0, 1) # (768, 2000)
        
        # Assuming Wᵤᵢ is a parameter of a model and needs to be frozen
        self.Wᵤᵢ.requires_grad = False

        self.n = n
        self.β = β
        # self.Sₐᵤ = nn.Parameter(torch.Tensor(Wᵤᵢ.size(0), out_features))
        
        # original
        #self.Sₐᵤ = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        self.Sₐᵤ = LinearFAModule(Wᵤᵢ.size(0), out_features,bias=False) # append bias term 
        
    def forward(self, vᵢ):
        # vᵢ = vᵢ.view(-1, 28, 28).transpose(1, 2).contiguous().view(-1, 28*28) # change vᵢ to be HxW for testing
        Wᵤᵢvᵢ = torch.matmul(vᵢ, self.Wᵤᵢ)
        hᵤ = F.relu(Wᵤᵢvᵢ) ** self.n
        Sₐᵤhᵤ = self.Sₐᵤ(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤhᵤ)
        return cₐ

class BioCell3_withoutFA(nn.Module):
    # Wᵤᵢ is the unsupervised pretrained weight matrix of shape: (2000, 28*28)
    def __init__(self, Wᵤᵢ, n=4.5, β=.01, out_features=10):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ.transpose(0, 1) # (768, 2000)
        
        # Assuming Wᵤᵢ is a parameter of a model and needs to be frozen
        self.Wᵤᵢ.requires_grad = False

        self.n = n
        self.β = β
        # self.Sₐᵤ = nn.Parameter(torch.Tensor(Wᵤᵢ.size(0), out_features))
        
        # original
        self.Sₐᵤ = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        #self.Sₐᵤ = LinearFAModule(Wᵤᵢ.size(0), out_features,bias=False) # append bias term 
        
    def forward(self, vᵢ):
        # vᵢ = vᵢ.view(-1, 28, 28).transpose(1, 2).contiguous().view(-1, 28*28) # change vᵢ to be HxW for testing
        Wᵤᵢvᵢ = torch.matmul(vᵢ, self.Wᵤᵢ)
        hᵤ = F.relu(Wᵤᵢvᵢ) ** self.n
        Sₐᵤhᵤ = self.Sₐᵤ(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤhᵤ)
        return cₐ
    
class BioLoss(nn.Module):
    def __init__(self, m=6):
        super().__init__()
        self.m = m

    def forward(self, cₐ, tₐ): 
        tₐ_ohe = torch.eye(10, dtype=torch.float, device='cuda')[tₐ]
        tₐ_ohe[tₐ_ohe==0] = -1.        
        loss = (cₐ - tₐ_ohe).abs() ** self.m
        return loss.sum()


class BioClassifier(nn.Module):
    def __init__(self, bio):
        super().__init__()
        self.bio = bio

    def forward(self, vᵢ):
        cₐ = self.bio(vᵢ)
        return F.log_softmax(cₐ, dim=-1)

class SimpleBioClassifier(nn.Module):
    def __init__(self, Wᵤᵢ, out_features):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ
        self.out = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ):
        Wᵤᵢvᵢ = F.linear(vᵢ, self.Wᵤᵢ, None)        
        return  F.log_softmax(self.out(Wᵤᵢvᵢ), dim=-1)

class BioConvLayer(nn.Module):
    def __init__(self, Wᵤᵢ):
        super().__init__()

        self.conv = nn.Conv2d(1, 2000, 28)
        self.conv.weight.data = Wᵤᵢ.view((-1, 1, 28, 28))        
        self.conv.weight.requires_grad = False
        
    def forward(self, vᵢ):
        x = vᵢ.view(len(vᵢ), 1, 28, 28)
        x = self.conv(x)        
        return x

class BioConvClassifier(nn.Module):
    def __init__(self, Wᵤᵢ, out_features):
        super().__init__()
        self.conv = BioConvLayer(Wᵤᵢ)
        self.out = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ):
        x = self.conv(vᵢ)
        x = x.view(len(vᵢ), -1)
        x = self.out(x)
        return x
        

# bio + FCN classifier - FCN with feedback alignment
class BioConvClassifier2(nn.Module):
    def __init__(self, Wᵤᵢ):
        super().__init__()
        self.conv1 = BioConvLayer(Wᵤᵢ)
        self.fc0 = LinearFAModule(2000, 256, bias=False)
        #self.conv2 = nn.Conv2d(2000, 256, 1)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(256, 1024)
        # self.fc2 = nn.Linear(1024, 10)
        self.fc1 = LinearFAModule(256, 1024, bias=False)
        self.fc2 = LinearFAModule(1024, 10, bias=False)

    def forward(self, x): # 64, 784
        x = self.conv1(x) # 64, 2000, 1, 1
        x = F.relu(x) # 64, 2000, 1, 1
        x = self.fc0(x) # 64, 256, 1, 1
        x = F.relu(x) # 64, 256, 1, 1
        x = x.squeeze() # 64, 256
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        # Use the custom LinearFAModule for forward pass
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# bio + FCN classifier - FCN with feedback alignment
class BioFCN_FA_Classifier(nn.Module):
    def __init__(self, Wᵤᵢ):
        super().__init__()
        self.conv1 = BioConvLayer(Wᵤᵢ)
        self.fc0 = LinearFAModule(2000, 256)
        self.fc1 = LinearFAModule(256, 1024)
        self.fc2 = LinearFAModule(1024, 10)

    def forward(self, x):  # 64, 784 
        x = self.conv1(x)  # 64, 2000, 1, 1
        x = F.relu(x)  # 64, 2000, 1, 1
        x = x.view(x.size(0), -1)  # Flatten the output to 2D
        x = self.fc0(x)  # 64, 256
        x = F.relu(x)  # 64, 256
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
    
class BioFCN_without_FA_Classifier(nn.Module):
    def __init__(self, Wᵤᵢ):
        super().__init__()
        self.conv1 = BioConvLayer(Wᵤᵢ)
        self.fc0 = nn.Linear(2000, 256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):  # 64, 784 
        x = self.conv1(x)  # 64, 2000, 1, 1
        x = F.relu(x)  # 64, 2000, 1, 1
        x = x.view(x.size(0), -1)  # Flatten the output to 2D
        x = self.fc0(x)  # 64, 256
        x = F.relu(x)  # 64, 256
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)