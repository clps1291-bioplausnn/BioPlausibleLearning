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


def train_semihebb(model, train_loader, test_loader, optimizer, loss_func, epochs, tsne_enabled=False):
    plt.ion()  # Interactive mode on

    accuracies = []
    for epoch in range(epochs):
        model.train()
        model.set_require_hebb(True)
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(b_x.size(0), -1)
            #print('before forward:')
            #print_model_weights(model=model)
            output = model(b_x)[0]   # output in shape of (50,10)
            #print('after forward:')
            #print_model_weights(model=model)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('after bp:')
            #print_model_weights(model=model)


            if step % 50 == 0:
                model.eval()
                model.set_require_hebb(False)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for test_x, test_y in test_loader:
                        test_x = test_x.view(test_x.size(0), -1)
                        test_output, last_layer = model(test_x)
                        pred_y = torch.max(test_output, 1)[1]
                        correct += (pred_y == test_y).sum().item()
                        total += test_y.size(0)

                    accuracy = 100 * correct / total
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Accuracy: {accuracy}%')
                    accuracies.append(round(accuracy/100,2))

                    if tsne_enabled:
                        # Visualization of trained flatten layer (T-SNE)
                        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                        #plot_only = 500
                        plot_only = min(500, last_layer.size(0))
                        low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                        labels = test_y.numpy()[:plot_only]
                        plot_with_labels(low_dim_embs, labels)

    plt.ioff()
    return accuracies








