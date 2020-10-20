# Image Classification on Fashion-MNIST
A simple Comparative Study for Image Classification on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset based on Neural Network algorithms.

A **short Report** explaining the workflow and architecture of the developed methods can be read [here](https://github.com/stevejpapad/Image-Classification-Fashion-MNIST/blob/main/Report.pdf).

### Approaches ###

1) Multi-Layer Perceptron (MLP)
2) Convolutional Neural Network (CNN)
3) Deep VGG-CNNs Pre-Trained on the Imagenet dataset
4) Image Augmentation and CNNs

```The code for each method can be found in:``` [**ImageClassification.py**](https://github.com/stevejpapad/Image-Classification-Fashion-MNIST/blob/main/ImageClassification.py)

### Experiments ###

1) MLP with one hidden layer (MLP1) 
2) MLP with two hidden layers (MLP2)
3) CNN with one hidden layer (CNN1) 
4) CNN with two hidden layers (CNN2) 
5) CNN with three hidden layers (CNN3)
6) Image augmentation + CNN1 (A-CNN1) 
7) Image augmentation + CNN2 (A-CNN2) 
8) Image augmentation + CNN3 (A-CNN3)
9) VGG16 pre-trained on ImageNet 
10) VGG19 pre-trained on ImageNet  

```The experiments can be run sequentially from:``` [**Main.py**](https://github.com/stevejpapad/Image-Classification-Fashion-MNIST/blob/main/Main.py)

### Results ###
The results in terms of Accuracy, for Training, Validation and Testing can be seen in the following figure 

![alt text](https://raw.githubusercontent.com/stevejpapad/Image-Classification-Fashion-MNIST/main/Results.png)

### Requirements ###
Tensorflow, Keras, Matplotlib, Sklearn, Graphviz (optional), Plotly (optional)
