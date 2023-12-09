import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.neural_networks import *
from data.load_data import *
from utils.others import *
from utils.plot import *
from utils.save_model import *


def train_fcn(model, train_loader, test_loader, optimizer, loss_func, epochs):
    plt.ion()  # Interactive mode on

    for epoch in range(epochs):
        model.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(b_x.size(0), -1) # flatten for fcn
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for test_x, test_y in test_loader:
                        test_x = test_x.view(test_x.size(0), -1)
                        test_output = model(test_x)
                        pred_y = torch.max(test_output, 1)[1]
                        correct += (pred_y == test_y).sum().item()
                        total += test_y.size(0)

                    accuracy = 100 * correct / total
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Accuracy: {accuracy}%')


    plt.ioff()  



#TODO:
# config, command-line arguments to set hyperparameters