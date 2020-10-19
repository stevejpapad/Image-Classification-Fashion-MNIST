import ImageClassification

''' 4 different approaches : 1) Multi-Layer Perceptron (MLP)
                             2) Convolutional Neural Network (CNN)
                             3) Deep VGG-CNN Pre-Trained on the Imagenet dataset
                             4) Image Augmentation and CNN
10 Experiments in total. Can be run sequentially. 
Requirements : Tensorflow, Keras, Matplotlib, Sklearn, Graphviz (optional), Plotly (optional)'''

# Simple MLP Model
ImageClassification.baseline_model(depth=1)
ImageClassification.baseline_model(depth=2)

# Convolutional Neural Network
# conv_depth defines the depth of the CNN. Can be from 1 to 3
ImageClassification.conv_nn(conv_depth=1)
ImageClassification.conv_nn(conv_depth=2)
ImageClassification.conv_nn(conv_depth=3)

# Convolutional Neural Network with Image Augmentation
# conv_depth defines the depth of the CNN. Can be from 1 to 3
ImageClassification.image_augmentation(conv_depth=1)
ImageClassification.image_augmentation(conv_depth=2)
ImageClassification.image_augmentation(conv_depth=3)

# VGG Model trained on ImageNet. Choice between VGG 16 or 19
ImageClassification.vgg_cnn(version=16)
ImageClassification.vgg_cnn(version=19)

