import torch
from data.load_data import *

def test_accuracy(model, dataset, flatten_input=False):
    if dataset not in ['mnist', 'cifar10']:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")
    
    if dataset=='mnist':
        _,test_loader=load_mnist(50)
    else:
        _,test_loader=load_cifar10(50)
    model.eval()  # evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for data in test_loader:
            images, labels = data
            if flatten_input:
                images = images.view(images.size(0),-1)
            #outputs, _ = model(images)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only the first output if there are multiple outputs

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the {model.__class__.__name__} on the {dataset.upper()} test images: {accuracy}%')

    accuracy = correct / total
    accuracy = round(accuracy, 3)

    return accuracy # [0,1], rounded to three decimal places
