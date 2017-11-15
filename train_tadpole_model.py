"""
An implementation of Convolutional Neural Network using TensorFlow
@author: Alexander Hamme
"""
from __future__ import print_function

from tensorflow.python.framework.errors_impl import InvalidArgumentError, OutOfRangeError
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib
import tempfile
import tflearn
import random
import pickle
import math
import cv2
import os

"""
TODO: Need to assemble (create) more images of tadpoles, training image dataset is far too small
"""

class TadpoleConvNet:

    MODEL_NAME = 'xenopus-tadpole-convnet'
    SAVEDATA = True
    IMG_SIZE = 28
    IMG_PIXELS = 784
    GRAYSCALE = True
    NUM_CHANNELS = 1                                # images will be converted to grayscale
    LR = 0.001                                      # learning rate
    STDDEV = 0.05                                   # use normal distribution with small variance for initial values of weights

    VAL_SPLIT = 0.2                                 # Validation split (number of images to hold back to test on)
    BATCH_SIZE = 16

    NUM_CLASSES = 2
    CLASS_ONE = "tadpole"
    CLASS_TWO = "ant"       # I chose ants as a class to train against, because ants look somewhat (but not too) similar to tadpoles

    def __init__(self):#, img_size):
        self.train_data = []#np.array([])
        self.test_data = []#np.array([])

    def create_label(self, image_name):
        """ Create a one-hot encoded vector from image name"""
        word_label = image_name.split('.')[0]

        if self.CLASS_ONE in word_label:
            return tf.convert_to_tensor([1, 0])  # , 0])
        elif self.CLASS_TWO in word_label:
            return tf.convert_to_tensor([0, 1])  # , 1])

    def create_train_data(self, train_dir, savename):

        """

        TODO: add image distortions to increase size of dataset

        :param train_dir: String, filename to either load, or save name for loaded data as np file
        :return: Tensor array

        """

        # save_data_imgs = []
        # save_data_lbls = []

        training_data = []
        labels = []

        prog_bar = tqdm(os.listdir(train_dir))             # Use tqdm for image loading progress meter
        image_reader = tf.WholeFileReader()

        for img_name in prog_bar:
            prog_bar.set_description("Loading training set")

            path = os.path.join(train_dir, img_name)

            if ".jpg" in path:
                img = tf.read_file(path)
                img_data = tf.image.decode_jpeg(img, channels=self.NUM_CHANNELS)

            elif ".png" in path:
                img = tf.read_file(path)
                img_data = tf.image.decode_png(img, channels=self.NUM_CHANNELS)
            else:
                print("Could not open image {}".format(path))
                continue

            img_data = tf.image.resize_images(img_data, [self.IMG_SIZE, self.IMG_SIZE])

            if self.GRAYSCALE:
                img_data = tf.image.rgb_to_grayscale(img_data)

            # img_data = tf.matmul(img_data, [1/255.0])

            training_data.append(img_data)
            labels.append(self.create_label(img_name))

        # random.shuffle(training_data)     not unless zip labels and imgs up together

        '''
        idx = self.VAL_SPLIT * len(training_data)
        train, val = training_data[:-idx], training_data[-idx:]

        train_dataset = tf.data.Dataset.from_tensor_slices((train, labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val, labels))
        '''

        # training_data = tf.data.Dataset.from_tensor_slices((training_data, labels))

        # train_dataset = tf.data.Dataset.from_tensor_slices((training_data, labels))
        # train_dataset.cache(filename=train_dir)
        # assert isinstance(train_dataset, tf.data.Dataset)

        """
        TODO: save as numpy file instead
        if self.SAVEDATA:
            with open(savename, "wb") as handle:
                pickle.dump([training_data, labels], handle)#, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(val_savename, "wb") as handle:
            #     pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        return training_data, labels  # train_dataset, val_dataset

    def create_test_data(self, test_dir, save_filename):

        testing_data = []
        labels = []

        prog_bar = tqdm(os.listdir(test_dir))  # Use tqdm for image loading progress meter

        for img_name in prog_bar:
            prog_bar.set_description("Loading testing set")

            path = os.path.join(test_dir, img_name)

            if ".jpg" in path:
                img = tf.read_file(path)
                img_data = tf.image.decode_jpeg(img, channels=self.NUM_CHANNELS)
            elif ".png" in path:
                img = tf.read_file(path)
                img_data = tf.image.decode_png(img, channels=self.NUM_CHANNELS)
            else:
                print("Could not open image {}".format(path))
                continue

            img_data = tf.image.resize_images(img_data, [self.IMG_SIZE, self.IMG_SIZE])

            if self.GRAYSCALE:
                img_data = tf.image.rgb_to_grayscale(img_data)

            # img_data = tf.matmul(img_data, [1/255.0])

            testing_data.append(img_data)
            labels.append(self.create_label(img_name))

        # testing_data = tf.data.Dataset.from_tensor_slices((testing_data, labels))

        # assert isinstance(testing_data, tf.data.Dataset)
        """TODO: save as numpy file instead
        if self.SAVEDATA:
            with open(save_filename, "wb") as handle:
                pickle.dump([testing_data, labels], handle)#protocol=pickle.HIGHEST_PROTOCOL)
        """
        return testing_data, labels

    def load_data(self, train, test, savetrain, savetest):
        # Check if dataset already exists
        if not (os.path.exists(savetrain) and os.path.exists(savetest)):
            # load data and save with given file names
            self.train_data = self.create_train_data(train, savetrain)
            self.test_data = self.create_test_data(test, savetest)
        else:
            self.train_data = pickle.load(savetrain)
            self.test_data = pickle.load(savetest)

    def weight_variable(self, shape, name):
        """
        To create this model, need to create a lot of weights and biases.
        Weights should be initialized with a small amount of noise for symmetry breaking, and to prevent 0 gradients.
        Since I'm implementing the Rectified Linear Units (ReLU) activation function, it's also good to create them 
        with a slightly positive initial bias, to avoid dead neurons.
        :param shape:
        :return:
        """
        std_dev = 1.0 / math.sqrt(float(TadpoleConvNet.IMG_PIXELS)) # todo:  or just 0.1?
        initial = tf.truncated_normal(shape, stddev=std_dev, name=name)           
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        # generates a bias variable of the given shape.
        initial = tf.constant(value=0.1, shape=shape, name=name)
        return tf.Variable(initial)

    def maxpool2d(self, tensor, k=2):
        # max_pool wrapper
        return tf.nn.max_pool(tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv2d(self, x, w, strides=1):
        # use full stride by default
        return tf.nn.conv2d(x, filter=w, strides=[1, strides, strides, 1], padding='SAME')

    def build_network(self, data):

        # if input Tensor is 1-D, reshape
        # with tf.name_scope('reshape'):            
        #     data = tf.reshape(data, [-1, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS])

        layer1_size = 32
        layer1_kernel = 5
        layer2_size = 64
        '''
        todo: test accuracy of deeper network after adding more hidden layers
        '''
        fc1_size = 1024

        # Hidden 1
        with tf.name_scope('conv1'):                                    # 1 image       or 1 channel?
            weights1 = self.weight_variable([layer1_kernel, layer1_kernel, 1, layer1_size], name='weights1')       # TadpoleConvNet.IMG_PIXELS
            biases1 = self.bias_variable([layer1_size], name='biases1')      # initialize biases as a Tensor array
            layer = self.conv2d(data, weights1) + biases1
            hidden1 = tf.nn.relu(layer)          # apply ReLU activation function

        # Max pooling layer 1 -- downsample by 2x2
        with tf.name_scope('maxpool1'):
            pool1 = self.maxpool2d(hidden1)

        # Hidden 2 -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            weights2 = self.weight_variable([layer1_kernel, layer1_kernel, layer1_size, layer2_size], name='weights2')    # todo: should this be (strides, strides) or (strides, filters)?
            biases2 = self.bias_variable([layer2_size], name='biases2')     # tf.zeros([layer2_size])
            layer2 = self.conv2d(pool1, weights2) + biases2
            hidden2 = tf.nn.relu(layer2)

        # Second pooling layer -- downsample again by 2x2
        with tf.name_scope('maxpool2'):
            pool2 = self.maxpool2d(hidden2)

        # Fully connected layer 1 -- after 2 round of downsampling, our NxN image
        # is down to N/4 x N/4 x 64 feature maps. This maps those to 1024 features.
        '''
        Image size has been reduced to N/4 X N/4, now add a fully-connected layer 
        with 1024 neurons to allow processing on the entire image. 
           
        Reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, 
        add bias, and apply ReLU.
        '''
        with tf.name_scope('fc1'):

            '''Note: this currently rests on the assumption that pooling has used 2X2 stride filter'''
            # todo: fix this to check what stride/filter kernel was used for pooling
            d_sampd_dim = TadpoleConvNet.IMG_SIZE / 2 / 2       # resulting dimension after two 2x2 max pool downsamplings

            weights3 = self.weight_variable([d_sampd_dim * d_sampd_dim * layer2_size, fc1_size], name='fc1_weights')
            biases3 = self.bias_variable([fc1_size], name='fc1_biases')

            '''Note: IMG_SIZE must (currently) be a multiple of 4 or else the round-off of integer division will cause an error'''
            flattened = tf.reshape(pool2, [-1, d_sampd_dim * d_sampd_dim * layer2_size])

            layer3 = tf.matmul(flattened, weights3) + biases3       # multiply features by weights vector and add biases

            fc_layer1 = tf.nn.relu(layer3)          # apply ReLU

        # Dropout - controls the complexity of the model by preventing co-adaptation of features and reducing overfitting.
        with tf.name_scope('dropout'):
            ''' 
            Apply dropout before the readout layer to reduce overfitting. 
            keep_prob is the probability that any single neuron's output is kept during dropout. 
            
            By doing this, dropout can be turned on during training and off during testing. 
            
            Also, the tf.nn.dropout op automatically handles scaling neuron outputs, so dropout doesn't need any additional scaling.
            '''
            keep_prob = tf.placeholder(tf.float32)
            fc_layer1_dropped = tf.nn.dropout(fc_layer1, keep_prob)

        # Map the 1024 features to NUM_CLASSES
        with tf.name_scope('fc2'):
            weights4 = self.weight_variable([fc1_size, self.NUM_CLASSES], name='fc2_weights')
            biases4 = self.bias_variable([self.NUM_CLASSES], name='fc2_biases')

            y_conv = tf.matmul(fc_layer1_dropped, weights4) + biases4

        return y_conv, keep_prob

    def run(self, iterations, batch_size, log_step=10):

        # x = tf.placeholder(tf.float32, [None, self.IMG_PIXELS])

        # initialize first layer for input
        x_imgs = tf.placeholder(tf.float32, shape=[None, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS], name='x_imgs')  # images
        y_lbls = tf.placeholder(tf.float32, shape=[None, self.NUM_CLASSES], name='y_lbls')  # labels

        y_conv, keep_prob = self.build_network(x_imgs)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_lbls, logits=y_conv)

        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            training_step = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            prediction = tf.argmax(y_conv, 1)
            truth = tf.argmax(y_lbls, 1)
            is_correct = tf.equal(prediction, truth)
            is_correct = tf.cast(is_correct, tf.float32)

        accuracy = tf.reduce_mean(is_correct)

        graph_loc = tempfile.mkdtemp()
        print("Saving graph to temp dir: {}".format(graph_loc))

        train_writer = tf.summary.FileWriter(graph_loc)
        train_writer.add_graph(tf.get_default_graph())

        '''
        training_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)     # tuple of img_data, labels
        # assert isinstance(training_dataset, tf.data.Dataset)
        testing_dataset = tf.data.Dataset.from_tensor_slices(self.test_data)
        blah = testing_dataset.batch(batch_size) 

        training_batch = tf.train.batch(self.train_data, batch_size, allow_smaller_final_batch=True)
        testing_batch = tf.train.batch(self.test_data, batch_size, allow_smaller_final_batch=True)
        '''

        # training_dataset = DataSet(self.train_data[0], self.train_data[1])
        # training_dataset.next_batch(batch_size=batch_size)
        # testing_dataset = DataSet(self.test_data[0], self.test_data[1])

        # training_data.repeat(count=3)
        # training_data.shuffle(...)

        # training_batch = tf.train.batch(self.train_data, batch_size, allow_smaller_final_batch=True)
        # testing_batch = tf.train.batch(self.test_data, batch_size, allow_smaller_final_batch=True)

        train_img_batch, train_lbl_batch = tf.train.batch(self.train_data, batch_size=batch_size)# ,num_threads=1)
        test_img_batch, test_lbl_batch = tf.train.batch(self.test_data, batch_size=batch_size)  # ,num_threads=1)

        # train_img_batch.

        training_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)     # tuple of img_data, labels
        # assert isinstance(training_dataset, tf.data.Dataset)
        # iterator = training_dataset.make_one_shot_iterator()

        saver = tf.train.Saver()

        iterator = training_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        '''testing_dataset = tf.data.Dataset.from_tensor_slices(self.test_data)'''

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            for i in range(iterations):
                # get next batch
                # batch = training_dataset#.next_batch(batch_size)  #.batch(batch_size)
                img_batch = []
                labels_batch = []
                for j in range(batch_size):
                    try:
                        pair = sess.run(next_element)
                    except OutOfRangeError as ignored:
                        print("reinitializing iterator")
                        sess.run(iterator.initializer)
                    else:
                        img_batch.append(pair[0])
                        labels_batch.append(pair[1])

                if i % log_step == 0:
                    train_accuracy = accuracy.eval(feed_dict={x_imgs: img_batch, y_lbls: labels_batch, keep_prob: 1.0}) #{x_imgs: batch[0], y_lbls: batch[1], keep_prob: 1.0})
                    print('step {}, training accuracy {}'.format(i, train_accuracy))

                training_step.run(feed_dict={x_imgs: img_batch, y_lbls: labels_batch, keep_prob: 0.5})#{x_imgs: batch[0], y_lbls: batch[1], keep_prob: 0.5})

            print("Shapes:", x_imgs.shape, y_lbls.shape)

            print('test accuracy {}'.format(        # \/  are these not the right dimensions?
                accuracy.eval(feed_dict={x_imgs: self.test_data[0], y_lbls: np.array(self.test_data[1]), keep_prob: 1.0}))
            )

            # saver.save(sess, '/saved/discriminator.ckpt')

    def create_weights(self, shape):
        # Create tf variable for weights, initialize values with low variance and stddev
        return tf.Variable(tf.truncated_normal(shape, stddev=self.STDDEV))

    def create_biases(self, size):
        # Create tf variable for biases, initialize values with low variance and stddev
        return tf.Variable(tf.constant(self.STDDEV, shape=[size]))

    def create_convolutional_layer(self, input_tensor, num_channels, filter_size, num_filters):
        """
        Creates convolutional layer, performs max pooling and ReLU activation before returning new Tensor
        :param input_tensor:    tf tensor  of  shape [batch, in_height, in_width, in_channels]
        :param num_channels:    image color channels, 3 if image is in RGB or 1 if it is monochrome
        :param filter_size:     kernel for filter size to extract subregions from image
        :param num_filters:     (number of neurons)
        :return:                2D tensor from 4D input and filter Tensors
        """

        # Initialize values for weights and biases with normal distribution and small variance
        weights = self.create_weights(shape=[filter_size, filter_size, num_channels, num_filters])  # Justify this
        biases = self.create_biases(num_filters)

        strides = [1, 1, 1, 1]        # [batch_stride, x_stride, y_stride, depth_stride].
        # - batch_stride must be 1, or it will skip images in the batch
        # - x_stride and y_stride are usually the same. In this case I used 1  TODO: test 2, check for improvement

        # Padding = 'SAME' means that we pad the input with zeros so that output x,y dimensions are same as input
        layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding='SAME',
                             use_cudnn_on_gpu=False, data_format="NHWC")  # NHWC = [batch, height, width, channels], as opposed to NCHW

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

        return layer            # type --> 4D Tensor?

    def create_flattened_layer(self, layer):
        """

        :param layer: multidimensional (4D) Tensor
        :return: reshaped one-dimensional Tensor
        """

        # convert 4D Tensor to one dimension
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])   # -1 tells tf that this will be dynamically updated. This will be used for the batch size
        return layer

    def create_fc_layer(self, input_tensor, num_inputs, num_outputs, use_relu=True):
        """
        create fully connected ("Dense") layer. It may be desirable to add a ReLU, so <use_relu> parameter added
        :param input_tensor:    (most likely one-dimensional) Tensor
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

        model_loaded = True  # check for files
        model_name = ""

        test_images = np.array(self.test_data, dtype=np.uint8)
        test_images = test_images.astype('float32')
        test_images = np.multiply(test_images, 1.0 / 255.0)

        num_images = len(test_images)

        print("number test images = ", num_images)

        # The input to the network is shape [<None> image_size image_size num_channels].
        x_batch = test_images.reshape(num_images, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS)


        sess = tf.Session()
        sess.run(tf.initialize_all_variables())


        if model_loaded:
            pass

        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph(model_name)

        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))

        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        print(result)

    def testConvNet(self):

        '''
        This is a test of a non-custom convnet implementation using TFLearn wrapper library for Tensorflow,
        to test what accuracy can be achieved on my current images dataset.

        Currently achieves about 80-90% accuracy, which is acceptable,
        but for this particular classification problem it should be possible to get much higher than that.
        I would guess it can reach at least 95%, because Xenopus tadpoles have a very distinct pattern and shape.
        '''

        N_EPOCHS = 25
        # My training dataset is currently too small to run more epochs,
        # the convnet is probably overfitting already

        self.load_data('train_data.npy', 'test_data.npy')

        train = self.train_data[:-self.VAL_SIZE]
        val = self.train_data[-self.VAL_SIZE:]

        X_train = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y_train = [i[1] for i in train]
        X_test = np.array([i[0] for i in val]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y_test = [i[1] for i in val]

        tf.reset_default_graph()
        convnet = input_data(shape=[-1, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')
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

        model.fit({'input': X_train}, {'targets': y_train}, n_epoch=N_EPOCHS,
                  validation_set=({'input': X_test}, {'targets': y_test}),
                  snapshot_step=TadpoleConvNet.VAL_SPLIT, show_metric=True, run_id=self.MODEL_NAME)

        model.save("tadpole_model")

        # Test model on testing dataset

        fig = plt.figure()

        for num, data in enumerate(self.test_data[:30]):  # visualize classifications for the first 30 test images

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

        numb_right = 0
        numb_wrong = 0
        total_numb = 0

        # Run predictions on entire testing data set and calculate accuracy
        for data in self.test_data:

            clss = data[1][1]
            img_data = data[0]
            img_data = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
            predict = model.predict([img_data])[0]

            if np.argmax(predict) == clss:
                numb_right += 1
            else:
                numb_wrong += 1
            total_numb += 1

        print("\nClassified {} correcly out of {} --> accuracy {:.3f}".format(numb_right, total_numb, float(numb_right)/total_numb))


tadconv = TadpoleConvNet()
tadconv.load_data("images_dataset/train", "images_dataset/test", "train_data.pickle", "test_data.pickle")
tf.app.run(main=tadconv.run(iterations=120, batch_size=40, log_step=10))

# except:
#     print("Could not load images")
#     raise SystemExit()

# tadconv.testConvNet()

# model.save("tadpole_model")

'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''

"""
TRAIN_DIR = 'images_dataset/train'
TEST_DIR = 'images_dataset/test'
IMG_SIZE = 50                           # dimension to resize to
LR = 0.001                              # learning rate
MODEL_NAME = 'xenopus-tadpole-convnet'

VAL_SPLIT = 20

CLASS_ONE = "tadpole"
CLASS_TWO = "ant"

def create_label(image_name):
    ''' Create an one-hot encoded vector from image name '''
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

train = train_data[:-VAL_SPLIT]
test = train_data[-VAL_SPLIT:]


X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

'''             Fairly shallow net:
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
'''
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
"""
