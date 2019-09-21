# Neural Networks

### Sia Peng and Angi Li

## What we built

We built two different neural network models using different libraries.

The first one utilized TensorFlow's existing fashion_mnist dataset to classify 10 different categories of clothing. We used Jupyter Notebook initially to run the model, then created a python file. We started out with a single layer perceptron model which gave us a validation accuracy rate of 85%, and then we improved the accuracy to 86% by implementing wider and deeper hidden layers in the network. Then, we further improved accuracy by restructuring the architecture to use Convolutional Neural Networks, which work better with images by analyzing groups of pixels to detect patterns in the image.

In the second experiment, we built a model to categorize images as either cats or dogs. We used a [dataset](https://www.kaggle.com/c/dogs-vs-cats/data) outside of the TensorFlow library that consisted of 30,000 images of cats and dogs. This program also uses a Convolutional Neural Network but with more hidden layers and larger numbers of neurons in each layer. This program is more useful than the first because since the data is in image form, we can use the model to predict any image we find of a cat and dog.

## Who did What

We both started out by researching neural networks--learning about what they were, how they were constructed, and the higher-level concepts behind why they work and how to adjust them to fit different types of data.

### Angi
I implemented the single perceptron version of the fashion neural network, as well as the first version of the CNN cat vs dog network.

### Sia

## What we learned

For constructing our first neural network, we use the Keras library which abstracts out a lot of the detailed math, which was nice since neither Sia nor I have worked with neural nets before. This was a good introduction to the higher level concept of neural nets, since we didn't get bogged down in the detailed math and tensor manipulations.

## What didn't work
