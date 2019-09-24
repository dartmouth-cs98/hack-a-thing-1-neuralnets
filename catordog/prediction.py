import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

'exec(%matplotlib inline)'

TRAIN_DIR = 'train'
TEST_DIR = 'test'
DEMO_DIR = 'demo'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogs-vs-cats-convnet'

# Creates a one-hot encoded vector from image name
# based on if image is a cat or dog
def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])

# Creates the training data by taking the images in /train,
# converting them to grayscale, resizing to 50, and adding
# the image data to the training dataset
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

# Same as above function but for the test data
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

def create_demo_data():
    demo_data = []
    for img in tqdm(os.listdir(DEMO_DIR)):
        path = os.path.join(DEMO_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        demo_data.append([np.array(img_data), img_num])
    shuffle(demo_data)
    np.save('demo_data.npy', demo_data)
    return demo_data

# train_data = create_train_data()
# test_data = create_test_data()
#demo_data = create_demo_data()
# If already created the dataset:
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
demo_data = np.load('demo_data.npy')

# Splitting the data into 24,500 for training and 500 for testing
train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

# Building the Convolutional Neural Network

tf.reset_default_graph()

# Input layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# Hidden layer with 32 neurons, stride of 5 (5x5 pixels)
convnet = conv_2d(convnet, 32, 5, activation='relu')
# Maxpool layer with 64 filters added
convnet = max_pool_2d(convnet, 5)

# Expand the model with more convolutional and max pool layers
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# Fully connected layer with 1024 neurons added
convnet = fully_connected(convnet, 1024, activation='relu')

# Dropout layer with a keep probability of 0.8 added
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

# Load models
model.load('catordog.model')

# -------------- Test --------------
# 1.Predict the first picture in the test_data set to test model accuracy
# d = test_data[0]
# img_data, img_num = d
#
# data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
# prediction = model.predict([data])[0]
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# ax.imshow(img_data, cmap="gray")
# print(f"cat: {prediction[0]}, dog: {prediction[1]}")
#
# 2. Predict the first 16 images using matplotlib plt
fig=plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='Dog'
    else:
        str_label='Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

# 3. Predict newly downloaded image from Google
# d = demo_data[0]
# img_data, img_num = d
#
# data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
# prediction = model.predict([data])[0]
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# ax.imshow(img_data, cmap="gray")
# print(f"cat: {prediction[0]}, dog: {prediction[1]}")
# plt.show()
