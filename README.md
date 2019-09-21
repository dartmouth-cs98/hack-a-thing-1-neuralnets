# Neural Networks

### Sia Peng and Angi Li

## What we built

We are both excited about using neuro networks in our project but neither of us has ever taken a ML class. Therefore, we experimented with neuro networks libraries in this assignment and built two different neural network models.

The first one utilized TensorFlow's existing fashion_mnist dataset to classify 10 different categories of clothing. We used Jupyter Notebook initially to run the model, then created a python file. We started out with a single layer perceptron model which gave us a validation accuracy rate of 85%, and then we improved the accuracy to 86% by implementing wider and deeper hidden layers in the network. Then, we further improved accuracy by restructuring the architecture to use Convolutional Neural Networks, which work better with images by analyzing groups of pixels to detect patterns in the image and improved accuracy rate to 90%.

![CNN](/img/cnn_fashionmist.jpg)

In the second experiment, we built a model to categorize images as either cats or dogs. We used a [dataset](https://www.kaggle.com/c/dogs-vs-cats/data) outside of the TensorFlow library that consisted of 30,000 images of cats and dogs. This program also uses a Convolutional Neural Network but with more hidden layers and larger numbers of neurons in each layer. This program is more useful than the first because since the data is in image form, we can use the model to predict any image we find of a cat and dog. Below is the result visualization of the first 16 images from the original test data:

![TEST](/img/catordog_run1.png)

We also tested this new model with images outside the existing dataset: we downloaded a cat image from Google and the model predicted it correctly.
Original Image:
![RAW](/img/cat1_test.jpg)
Result:
![VIS](/img/cat1_raw.jpg)
![RESULT](/img/cat1_result.jpg)


## Who did What

We both started out by researching neural networks--learning about what they were, how they were constructed, and the higher-level concepts behind why they work and how to adjust them to fit different types of data.

### Angi
I implemented the single perceptron version of the fashion neural network, as well as the first version of the CNN cat vs dog network.

### Sia
I implemented the CNN version of the fashion neural network. For the cat vs dog network, I expanded the CNN model with more layers to improve validation accuracy. I also implemented prediction.py to test the saved model with selected images and matplotlib plt visualization.

## What we learned

For constructing our first neural network, we use the Keras library which abstracts out a lot of the detailed math, which was nice since neither Sia nor I have worked with neural nets before. This was a good introduction to the higher level concept of neural nets, since we didn't get bogged down in the detailed math and tensor manipulations.

We also learned different types of neuro network models, such as Perceptron and Convolutional Neural Networks. The CNN model can detect patterns in the image and is more accurate than the perceptron model. We also learned how to improce both types of networks by widening (#layers) and deeping(#neurons) hidden layers. Furthermore, we also experimented training and testing models with outside dataset, which can be applied to our future project.

## What didn't work
The second model only has a 80% accuracy, probably because we reduced the image size drastically and set a relatively low probability parameter of the Dropout layer and the optimizer in order to train faster. In this way, we prioritized speed over accuracy but we can potentially improving the model with better architecture.


## Reference
- [A step-by-step neural network tutorial for beginners] (https://becominghuman.ai/step-by-step-neural-network-tutorial-for-beginner-cc71a04eedeb)
- [Building a Cat Detector using Convolutional Neural Networks — TensorFlow for Hackers (Part III)] (https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)
