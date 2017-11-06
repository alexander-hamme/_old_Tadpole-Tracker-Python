import cv2
import matplotlib
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

# https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b

TRAIN_DIR = 'images_dataset/train'
TEST_DIR = 'images_dataset/test'
IMG_SIZE = 50                           # dimension to resize to
LR = 0.001
MODEL_NAME = 'xenopus-tadpole-convnet'

TEST_SIZE = 20


CLASS_ONE_NAME = "tadpole"
CLASS_TWO_NAME = "ant"

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]

    if CLASS_ONE_NAME in word_label:
        return np.array([1,0])
    elif CLASS_TWO_NAME in word_label:
        return np.array([0,1])


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


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)

        # img_num = img.split('.')[0]
        # print img_num

        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# Check if you have already created the dataset:
if not(os.path.exists('train_data.npy') or os.path.exists('test_data.npy')):
    train_data = create_train_data()
    test_data = create_test_data()
else:
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')


# print train_data[:10]
# print test_data[:10]

train = train_data[:-TEST_SIZE]
test = train_data[-TEST_SIZE:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

"""             Fairly shallow net:
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=TEST_SIZE, show_metric=True, run_id=MODEL_NAME)
"""

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=20,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=TEST_SIZE, show_metric=True, run_id=MODEL_NAME)

fig = plt.figure()#figsize=(16, 12))

# Test model

for num, data in enumerate(test_data[:30]):#[:16]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(5, 6, num + 1)          #nrows, ncols --> dependent on how many images you're showing
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = ''
    else:
        str_label = 'Tadpole'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

# model.save("tadpole_model")
