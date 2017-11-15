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

            # saver.save(sess, '/model/classifier.ckpt')

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
# tadconv.testConvNet()
# model.save("tadpole_model")

'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/images_dataset/train',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''
