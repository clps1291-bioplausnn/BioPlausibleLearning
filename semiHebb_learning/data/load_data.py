# data.py
import os
import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist(batch_size):
    DOWNLOAD_MNIST = False
    if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is an empty dir
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        download=DOWNLOAD_MNIST,
        transform=transforms.Compose(
            [transforms.ToTensor(), # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of (C,H,W) and normalize in the range [0.0, 1.0],
            transforms.Normalize([0.5], [0.5])]) # normalize to [-1,1] for faster convergence
    
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True
    )

    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]))
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_cifar10(batch_size):
    DOWNLOAD_CIFAR10 = False
    if not(os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
        DOWNLOAD_CIFAR10 = True

    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #for each RGB channel.
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='./cifar10/', train=True,
        download=DOWNLOAD_CIFAR10, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.CIFAR10(
        root='./cifar10/', train=False,
        download=DOWNLOAD_CIFAR10, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader