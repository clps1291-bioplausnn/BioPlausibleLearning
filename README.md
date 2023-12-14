# Biologically plausible neural networks

In the pursuit of artificial intelligence that mirrors the deftness of human cognition, the concept of biological plausibility stands as a beacon, guiding the design of neural networks toward the intricate workings of the human brain. A neural network that is considered biologically plausible emulates the structure and functions of the biological nervous system, often with the purpose of improving the performance of neural networks or gaining insights into processes of the biological brain. 

While backpropagation (BP) is a cornerstone in training modern neural networks, it deviates from how biological neural systems function. Key differences include BP's reliance on inter-layer weight dynamics, unlike the local information transmission in biological neurons, its use of symmetric weights for both forward and backward passes which contrasts with the one-directional, asymmetric nature of biological synapses, and its continuous output neuron firing, as opposed to the all-or-none firing based on a threshold in biological neurons. 
Recognizing these discrepancies, this project focuses on exploring neural network techniques that better mimic human brain functions. The aim is to investigate how these biologically inspired alternatives to backpropagation could enhance the performance and interpretability of neural networks. 

A full version report can be found here: [LINK TO PDF]

## Folder Explanation 

- **hybridBio_learning**: A PyTorch implementation of “Unsupervised learning by competing hidden units” MNIST classifier, combining with Feedback alignment. Original descriptive documentation can be found at [here](https://github.com/clps1291-bioplausnn/hybrid-bioLearning).

- **semiHebb_learning**

### Pretrain lighter vision models
Recognizing the need for more accessible alternatives to large pretrained vision models on imagenet, this repo aims to provide models pretrained on smaller datasets like MNIST and CIFAR10. These lighter and more manageable models are pretrained for easy import and utilization, facilitating quick experimentation and integration into projects where resources are limited. 

The MNIST database contains 60,000 training images and 10,000 testing images.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

### Learning rules

### Neural networks

### Evaluation



# Reference:

1. [PyTorch CIFAR10 by huyvnphan](https://github.com/huyvnphan/PyTorch_CIFAR10)

2. [MNIST_database](https://en.wikipedia.org/wiki/MNIST_database)

3. [Unsupervised Bio Classifier](https://github.com/gatapia/unsupervised_bio_classifier)

4. [Linear FA implementation](https://github.com/L0SG/feedback-alignment-pytorch)

Except for torchvision models, [GluonCV](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo) includes many sota models in CV.
