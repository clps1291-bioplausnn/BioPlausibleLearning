
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_example(data_loader):
    """
    Plots an example image and its label from a given DataLoader.
    Handles both grayscale and RGB images and undoes normalization.

    Args:
    data_loader (DataLoader): A PyTorch DataLoader from which to plot an example.
    """
    examples = iter(data_loader)
    example_data, example_targets = next(examples)

    np_image = example_data[0].numpy().transpose((1, 2, 0))
    np_image = np_image / 2 + 0.5  # reverse the normalization

    if example_data[0].shape[0] == 1:  # single color channel
        plt.imshow(np_image[:, :, 0], cmap='gray')
    else:  # RGB
        plt.imshow(np_image)

    plt.title(f'Label: {example_targets[0]}')
    plt.show()


# t-SNE visualization
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

def plot_score_comparison(x_axis,accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, accuracies, marker='o')

    plt.title('Model Accuracy Comparison')
    plt.xlabel('Ratio of Hebbian Layers (#HebbLayers/#Total Layers)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(x_axis)
    plt.ylim(0, 1) 

    plt.show()