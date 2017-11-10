"""
An implementation of Convolutional Neural Network using TensorFlow
@author: Alexander Hamme
"""
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib
import tflearn
import cv2
import os

# https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b

# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

# https://github.com/alexander-hamme/cv-tricks.com

# http://cv-tricks.com/

"""Architecture of the network

When designing the architecture of a neural network you have to decide on:
How do you arrange layers? which layers to use? how many neurons to use in each layer etc.? 
Designing the architecture is slightly complicated and advanced topic and takes a lot of research. 
"""
"""
TODO: Need to assemble more images of tadpoles, training image dataset is far too small
"""


class TadpoleConvNet:

    MODEL_NAME = 'xenopus-tadpole-convnet'
    TRAIN_DIR = 'images_dataset/train'
    TEST_DIR = 'images_dataset/test'
    IMG_SIZE = 50                                   # dimension to resize to
    NUM_CHANNELS = 1                                # images will be converted to grayscale
    LR = 0.001                                      # learning rate
    STDDEV = 0.05                                   # use normal distribution with small variance for initial values of weights

    VAL_SIZE = 20                                   # Validation size (number of images to hold back to test on)
    BATCH_SIZE = 16

    NUM_CLASSES = 2
    CLASS_ONE = "tadpole"
    CLASS_TWO = "ant"        # I chose ants as a class to train against, because ants look somewhat similar to tadpoles

    def __init__(self):
        self.train_data = np.array([])
        self.test_data = np.array([])

    def create_label(self, image_name):
        """ Create a one-hot encoded vector from image name"""
        word_label = image_name.split('.')[0]

        if self.CLASS_ONE in word_label:
            return np.array([1, 0])
        elif self.CLASS_TWO in word_label:
            return np.array([0, 1])

    def create_train_data(self, filename):

        training_data = []

        prog_bar = tqdm(os.listdir(self.TRAIN_DIR))

        for img in prog_bar:        # Use tqdm for image loading progress meter

            prog_bar.set_description("Loading training set")

            path = os.path.join(self.TRAIN_DIR, img)

            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            img_data = cv2.resize(img_data, (self.IMG_SIZE, self.IMG_SIZE))

            training_data.append([np.array(img_data), self.create_label(img)])

        shuffle(training_data)
        np.save(filename, training_data)
        return training_data

    def create_test_data(self, filename):

        testing_data = []

        prog_bar = tqdm(os.listdir(self.TEST_DIR))

        for img in prog_bar:

            prog_bar.set_description("Loading testing set")

            path = os.path.join(self.TEST_DIR, img)

            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (self.IMG_SIZE, self.IMG_SIZE))

            testing_data.append([np.array(img_data), self.create_label(img)])

        shuffle(testing_data)
        np.save(filename, testing_data)
        return testing_data

    def load_data(self, train, test):
        # Check if dataset already exists
        if not (os.path.exists(train) and os.path.exists(test)):
            self.train_data = self.create_train_data(train)
            self.test_data = self.create_test_data(test)
        else:
            self.train_data = np.load(train)
            self.test_data = np.load(test)

    def create_weights(self, shape):
        # Create tf variable for weights, initialize values with low variance and stddev
        return tf.Variable(tf.truncated_normal(shape, stddev=self.STDDEV))

    def create_biases(self, size):
        # Create tf variable for biases, initialize values with low variance and stddev
        return tf.Variable(tf.constant(self.STDDEV, shape=[size]))

    def create_convolutional_layer(self, input_tensor, num_channels, filter_size, num_filters):
        """
        Creates convolutional layer, performs max pooling and ReLU activation before returning new Tensor
        :param input_tensor:    tf tensor
        :param num_channels:    image color channels, 3 if image is in RGB or 1 if it is monochrome
        :param filter_size:     kernel for filter size to extract subregions from image
        :param num_filters:     (number of neurons)
        :return:                multidimensional (4D) tensor
        """

        # Initialize values for weights and biases with normal distribution and small variance
        weights = self.create_weights(shape=[filter_size, filter_size, num_channels, num_filters])  # Justify this
        biases = self.create_biases(num_filters)

        strides = [1, 1, 1, 1]        # [batch_stride, x_stride, y_stride, depth_stride].
        # - batch_stride must be 1, or it will skip images in the batch
        # - x_stride and y_stride are usually the same. In this case I used 1  TODO: test 2, check for improvement

        # Padding = 'SAME' means that we pad the input with 0s so that output x,y dimensions are same as input
        layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding='SAME')

        print(type(layer))

        layer += biases

        print(type(layer))

        filter_size = [1, 2, 2, 1]  # ksize / filter size are 2*2 in x and y direction, and 1 for batch and depth.
        ksize = [1, 2, 2, 1]        # Note that specifying stride size to be 2 ensures pooled regions do not overlap

        # perform max-pooling --> extract subregions of feature map, making output exactly half the size of the input
        layer = tf.nn.max_pool(value=layer, ksize=ksize, strides=filter_size, padding='SAME')

        print(type(layer))

        # Pass output of max pooling to ReLU, which is the   >>>activation function   ??
        layer = tf.nn.relu(layer)

        print(type(layer))

        return layer            # type --> multidimensional Tensor?

    def create_flattened_layer(self, layer):
        """

        :param layer: multidimensional (4D) Tensor
        :return: reshaped one dimensional Tensor
        """

        # convert 4D Tensor to one dimension
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])   # -1 tells tf that this will be dynamically updated. This will be for the batch size
        return layer

    def create_fc_layer(self, input_tensor, num_inputs, num_outputs, use_relu=True):
        """
        create fully connected ("Dense") layer. It may be desirable to add a ReLU, so <use_relu> parameter added
        :param input_tensor:    (most likely one dimensional) Tensor
        :param num_inputs:      number of incoming connections (from neurons)
        :param num_outputs:     number of outgoing connections
        :param use_relu:        whether to use ReLU activation
        :return: "dense" layer Tensor
        """
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)

        layer = tf.matmul(input_tensor, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)

        return layer


    def create_network(self):
        """

        this is still in progress.

        TODO: try out different architecture designs to see if any give improvement
        :return: accuracy or validation loss ?
        """

        filter_size = 5                         # 5x5 filters (extracting 5x5 pixel subregions) with ReLU

        layer1_size = 32                        # 32 of these filters (neurons)
        layer2_size = 64
        dense_lay1_size = 1024                  # the fully connected ("dense") layer of 1024 neurons
        dropout_rate = 0.4                      # dropout regularization rate

        # First dimension of shape is None so that any number of images can be passed to it
        x = tf.placeholder(tf.float32, shape=[None, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS], name='x')  # images
        y = tf.placeholder(tf.placeholder(tf.float32, shape=[None, self.NUM_CLASSES], name='y'))                 # labels

        conv_layer1 = self.create_convolutional_layer(
            input_tensor=x, num_channels=self.NUM_CHANNELS, filter_size=filter_size, num_filters=layer1_size
        )

        conv_layer2 = self.create_convolutional_layer(
            input_tensor=conv_layer1, num_channels=layer1_size, filter_size=filter_size, num_filters=layer2_size
        )
        '''
        Deeper structure: add three more layers here before the flattened and dense layers?
        --> 128, 64, 32
        '''

        flattened = self.create_flattened_layer(conv_layer2)
        flattened_outputs = flattened.get_shape()[1:4].num_elements()

        dense_layer1 = self.create_fc_layer(
            input_tensor=flattened, num_inputs=flattened_outputs, num_outputs=dense_lay1_size, use_relu=True
        )

        # TODO: add create_dropout_layer function?
        dropout_layer = tf.layers.dropout(
            inputs=dense_layer1, rate=dropout_rate, training=True
        )

        final_layer = self.create_fc_layer(
            input_tensor=dropout_layer, num_inputs=dense_lay1_size, num_outputs=self.NUM_CLASSES, use_relu=False
        )

        y_pred = tf.nn.softmax(final_layer, name="y_pred")      # will contain 2 predicted probabilities, one for each class

        predicted_class = tf.argmax(y_pred, dimension=1)        # class with higher probability is the network's prediction

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=y)
        cost = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(cost)


        # TODO





    def testConvNet(self):

        '''
        This is a test of a non-custom convnet implementation using Tensorflow,
        to test what accuracy can be achieved on my current images dataset
        '''

        self.load_data('train_data.npy', 'test_data.npy')

        train = self.train_data[:-self.VAL_SIZE]
        val = self.train_data[-self.VAL_SIZE:]

        X_train = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y_train = [i[1] for i in train]
        X_test = np.array([i[0] for i in val]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y_test = [i[1] for i in val]

        tf.reset_default_graph()
        convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')
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
        convnet = regression(convnet, optimizer='adam', learning_rate=self.LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

        model.fit({'input': X_train}, {'targets': y_train}, n_epoch=20,
                  validation_set=({'input': X_test}, {'targets': y_test}),
                  snapshot_step=self.VAL_SIZE, show_metric=True, run_id=self.MODEL_NAME)

        model.save("tadpole_model")

        # Test model on testing dataset

        fig = plt.figure()

        for num, data in enumerate(self.test_data[:30]):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(5, 6, num + 1)  # nrows, ncols --> dependent on how many images you're showing in the frame
            orig = img_data
            data = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 1:
                str_label = ''      # detecting tadpoles is the only classification of interest
            else:
                str_label = 'tadpole'

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

        plt.show()


tadconv = TadpoleConvNet()
tadconv.testConvNet()

raise SystemExit()


TRAIN_DIR = 'images_dataset/train'
TEST_DIR = 'images_dataset/test'
IMG_SIZE = 50                           # dimension to resize to
LR = 0.001                              # learning rate
MODEL_NAME = 'xenopus-tadpole-convnet'

VAL_SIZE = 20

CLASS_ONE = "tadpole"
CLASS_TWO = "ant"

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[0]

    if CLASS_ONE in word_label:
        return np.array([1,0])
    elif CLASS_TWO in word_label:
        return np.array([0,1])


def create_train_data():

    training_data = []

    images_folder = os.listdir(TRAIN_DIR)

    for img in images_folder:

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

train = train_data[:-VAL_SIZE]
test = train_data[-VAL_SIZE:]


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
#
# tf.reset_default_graph()
# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# convnet = fully_connected(convnet, 2, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#
# model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
#
#
# model.fit({'input': X_train}, {'targets': y_train}, n_epoch=20,
#           validation_set=({'input': X_test}, {'targets': y_test}),
#           snapshot_step=VAL_SIZE, show_metric=True, run_id=MODEL_NAME)
#
# model.save("tadpole_model")
#
# # Test model
#
# fig = plt.figure()#figsize=(16, 12))
#
# for num, data in enumerate(test_data[:30]):#[:16]):
#
#     img_num = data[1]
#     img_data = data[0]
#
#     y = fig.add_subplot(5, 6, num + 1)          #nrows, ncols --> dependent on how many images you're showing
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#     model_out = model.predict([data])[0]
#
#     if np.argmax(model_out) == 1:
#         str_label = ''
#     else:
#         str_label = 'Tadpole'
#
#     y.imshow(orig, cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
#
# plt.show()

# model.save("tadpole_model")
