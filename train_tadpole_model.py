"""
An implementation of a Convolutional Neural Network using TensorFlow
@author: Alexander Hamme
"""
from __future__ import print_function
from tensorflow.python.framework.errors_impl import InvalidArgumentError, OutOfRangeError
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tensorflow.python.data.util import nest
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib
import tempfile
import tflearn
import random
import pickle
import time
import math
import cv2
import sys
import os

class TadpoleConvNet:

    MODEL_NAME = 'xenopus-tadpole-convnet'
    IMG_SIZE = 28                                   # dimension to resize to. should be a multiple of 4.
    IMG_PIXELS = 784  # 28*28
    GRAYSCALE = True
    NUM_CHANNELS = 1                                # (xenopus tadpoles' colors are just black/grey/white anyway)
    LR = 0.001                                      # learning rate
    STDDEV = 0.05                                   # use normal distribution with small variance for initial weights

    VAL_SPLIT = 0.2                                 # Validation split (number of images to hold back to test on)
    BATCH_SIZE = 100
    MAKE_BATCHES = False                            # current dataset is not large enough to justify using batches

    NUM_CLASSES = 2
    CLASS_ONE = "tadpole"
    CLASS_TWO = "negative"                          # background class

    def __init__(self, model_save_name, session_save_name):
        self.train_data = []
        self.test_data = []
        self.save_data = True
        self.model_save = model_save_name
        self.session_save = session_save_name
        self.loss_data = []
        # self.img_size = img_size                    # dimension to resize to
        # self.img_pixels = img_size**2

    def create_label(self, image_name, save_type):
        """
        Create a one-hot encoded vector from image name
        Note that this current function is based on the
        assumption that there are only two classes
        :param typ: either "np" or "tf"
        :return: either a numpy array or a Tensor, depending on `typ`"""

        word_label = image_name.split('.')[0]

        if TadpoleConvNet.CLASS_ONE in word_label:
            return np.array([1, 0]) if save_type == "np" else tf.convert_to_tensor([1, 0])

        elif TadpoleConvNet.CLASS_TWO in word_label:
            return np.array([0, 1]) if save_type == "np" else tf.convert_to_tensor([0, 1])

    def create_train_data(self, train_dir, savename, save_type):

        """
        Create training data
        :param train_dir: String, path to images folder
        :return: array of Tensors or Numpy.ndarrays, depending on `save_type` parameter
        """

        #  TODO: add image contortions to increase dataset size and variance

        if save_type not in ("np", "tf"):
            raise ValueError("Invalid argument for 'save_type' parameter.\nValid options are 'np' or 'tf'.")

        training_data = []
        labels = []

        prog_bar = tqdm(os.listdir(train_dir))             # Use tqdm for image loading progress meter

        for img_name in prog_bar:

            prog_bar.set_description("Loading training set")
            path = os.path.join(train_dir, img_name)

            if save_type == "np":

                if TadpoleConvNet.GRAYSCALE:
                    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                else:
                    img_data = cv2.imread(path)

                if img_data is None:
                    print("Could not load image {}".format(path))
                    continue

                img_data = cv2.resize(img_data, (self.IMG_SIZE, self.IMG_SIZE))
                img_data = img_data[:, :, np.newaxis]                   # img dimensions must match input tensor dims

                # TODO: normalize the images??
                # image = image.astype(np.float32)
                # image = np.multiply(image, 1.0 / 255.0)

            else:
                if ".jpg" in path:
                    img = tf.read_file(path)
                    img_data = tf.image.decode_jpeg(img, channels=TadpoleConvNet.NUM_CHANNELS)

                elif ".png" in path:
                    img = tf.read_file(path)
                    img_data = tf.image.decode_png(img, channels=TadpoleConvNet.NUM_CHANNELS)
                else:
                    print("Could not open image {}".format(path))
                    continue

                # if    <<check if Tensor is empty>>
                #     print("Could not load image {}".format(path))
                #     continue

                img_data = tf.image.resize_images(img_data, [TadpoleConvNet.IMG_SIZE, TadpoleConvNet.IMG_SIZE])

                if TadpoleConvNet.GRAYSCALE:
                    img_data = tf.image.rgb_to_grayscale(img_data)

                # img_data = tf.matmul(img_data, [1/255.0])

            training_data.append(img_data)
            labels.append(self.create_label(img_name, save_type=save_type))

        if self.save_data and save_type == "np":
            # np.save(savename, (training_data, labels))
            np.savez(savename, training_data, labels)
        else:
            pass        # figure out saving list of Tensors to file  -- TFRecords format?

        return (training_data, labels)

    def create_test_data(self, test_dir, savename, save_type):

        testing_data = []
        labels = []

        prog_bar = tqdm(os.listdir(test_dir))  # Use tqdm for image loading progress meter

        for img_name in prog_bar:
            prog_bar.set_description("Loading testing set")

            path = os.path.join(test_dir, img_name)

            if save_type == "np":
                if self.GRAYSCALE:
                    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                else:
                    img_data = cv2.imread(path)

                if img_data is None:
                    print("Could not load image {}".format(path))
                    continue

                img_data = cv2.resize(img_data, (self.IMG_SIZE, self.IMG_SIZE))
                img_data = img_data[:, :, np.newaxis]                   # img dimensions must match input tensor dims

                # image = image.astype(np.float32)
                # image = np.multiply(image, 1.0 / 255.0)

            else:
                if ".jpg" in path:
                    img = tf.read_file(path)
                    img_data = tf.image.decode_jpeg(img, channels=self.NUM_CHANNELS)
                elif ".png" in path:
                    img = tf.read_file(path)
                    img_data = tf.image.decode_png(img, channels=self.NUM_CHANNELS)
                else:
                    print("Could not open image {}".format(path))
                    continue

                # if    <<check if Tensor is empty>>
                #     print("Could not load image {}".format(path))
                #     continue

                img_data = tf.image.resize_images(img_data, [self.IMG_SIZE, self.IMG_SIZE])

                if self.GRAYSCALE:
                    img_data = tf.image.rgb_to_grayscale(img_data)

                # img_data = tf.matmul(img_data, [1/255.0])

            testing_data.append(img_data)
            labels.append(self.create_label(img_name, save_type=save_type))

        if self.save_data and save_type == "np":
            # np.save(savename, (testing_data, labels))
            np.savez(savename, testing_data, labels)
            # with open(savename, "wb") as handle:
            #     pickle.dump((testing_data, labels), handle)
        else:
            pass

        return testing_data, labels

    def load_data(self, train, test, savetrain, savetest, save_type):
        # Check if dataset already exists
        savetrain = savetrain if '.npz' in savetrain else savetrain + '.npz'
        savetest = savetest if '.npz' in savetest else savetest + '.npz'

        if not (os.path.exists(savetrain) and os.path.exists(savetest)):
            # load data and save with given file names
            self.train_data = self.create_train_data(train, savetrain, save_type=save_type)
            self.test_data = self.create_test_data(test, savetest, save_type=save_type)
            print("Saved train and test data to: \n/{}\n/{}".format(savetrain, savetest))
        else:
            # load data with given save file names.
            # IOError should not need to be caught, because if this block is reached, the file exists.
            if save_type == "np":

                print("Loaded train and test data from: \n/{}\n/{}".format(savetrain, savetest))

                npzfile1 = np.load(savetrain)
                self.train_data = (npzfile1['arr_0'], npzfile1['arr_1'])  # type must be tuple of 2 lists of ndarrays

                npzfile2 = np.load(savetest)
                self.test_data = (npzfile2['arr_0'], npzfile2['arr_1'])

            else:
                # figure out serializing and loading list of Tensors
                pass

    def weight_variable(self, shape, name=None):
        """
        To create this model, need to create a lot of weights and biases.
        Weights should be initialized with a small amount of noise for symmetry breaking, and to prevent 0 gradients.
        Since I'm implementing the Rectified Linear Units (ReLU) activation function, it's also good to create them
        with a slightly positive initial bias, to avoid dead neurons.

        :param shape:
        :return:
        """
        std_dev = 1.0 / math.sqrt(float(TadpoleConvNet.IMG_PIXELS))     # todo:  or just a simple number like 0.05?
        initial = tf.truncated_normal(shape, stddev=std_dev, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, val=0.1, name=None):
        # generates a bias variable of the given shape, with 0.1 as default value.
        initial = tf.constant(value=val, shape=shape, name=name)
        return tf.Variable(initial)

    def maxpool2d(self, tensor, k=2, name=None):
        # max_pool wrapper with default stride of 2x2
        return tf.nn.max_pool(tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv2d(self, x, w, strides=1, name=None):
        # use full stride by default
        return tf.nn.conv2d(x, filter=w, strides=[1, strides, strides, 1], padding='SAME', name=name)

    def build_network(self, data):
        """
        :param data: list of Numpy arrays
        :return: the constructed conv_net and the keep_prob Tensorflow variable,
        which is the probability that each element is kept (used for dropout layers)
        """

        # todo: assert data.shape = ....... # Tensor/ndarray shape should be 4-D: [Batch Size, Height, Width, Channel]

        # if input Tensor is 1-D, reshape
        # with tf.name_scope('reshape'):
        #     data = tf.reshape(data, [-1, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNELS])

        # add variable for stride/filter kernel to be used for maxpooling
        layer1_size = 32
        layer1_kernel = 5
        layer2_size = 64
        # '''
        # TODO: test accuracy of deeper network, add more hidden layers
        # layer2_kernel = 3?  or should first kernel size be 3 and second one be 5?
        # '''
        fc1_size = 1024

        ''' It's good practice to give important variables names, in case you want to access them later on.
        
            Using name scoping makes it so that the resulting graph is easy to read    
        '''

        # Hidden layer 1
        with tf.name_scope('conv1'):                # \/ does this make sense to have NUM_CHANNELS here? << todo
            weights1 = self.weight_variable(
                [layer1_kernel, layer1_kernel, TadpoleConvNet.NUM_CHANNELS, layer1_size],
                name='weights1'
            )
            biases1 = self.bias_variable([layer1_size], name='biases1')  # initialize biases as an array Tensor
            layer = self.conv2d(data, weights1) + biases1
            hidden1 = tf.nn.relu(features=layer)                         # apply Rectified Linear activation function

        # Max pooling layer 1 -- downsample by 2x2 strides
        with tf.name_scope('maxpool1'):
            pool1 = self.maxpool2d(hidden1)

        # Hidden 2 -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            weights2 = self.weight_variable([layer1_kernel, layer1_kernel, layer1_size, layer2_size], name='weights2')    # todo: should this be (strides, strides) or (strides, filters)?
            biases2 = self.bias_variable([layer2_size], name='biases2')
            layer2 = self.conv2d(pool1, weights2) + biases2
            hidden2 = tf.nn.relu(features=layer2)

        # Second pooling layer -- downsample again by 2x2
        with tf.name_scope('maxpool2'):
            pool2 = self.maxpool2d(hidden2)

        # Fully connected ("dense") layer
        with tf.name_scope('fc1'):
            '''
            After 2 rounds of 2x2 downsampling, image size has been reduced to (N/4) x (N/4) x 64 feature maps.

            Add a fully-connected layer with 1024 neurons to allow processing on the entire image.

            To do this, reshape the tensor from the pooling layer into a batch of vectors,
            multiply by a weight matrix, add biases, and apply ReLU.
            '''

            # Note: this currently rests on the assumption that pooling has used 2x2 stride filter

            d_sampd_dim = TadpoleConvNet.IMG_SIZE / 2 / 2           # resulting dimension after two 2x2 downsamplings

            weights3 = self.weight_variable([d_sampd_dim * d_sampd_dim * layer2_size, fc1_size], name='fc1_weights')
            biases3 = self.bias_variable([fc1_size], name='fc1_biases')

            # Note: assuming maxpool uses 2x2 stride, IMG_SIZE must be a multiple of 4,
            # or the round-off of integer division here will cause an error
            flattened = tf.reshape(pool2, [-1, d_sampd_dim * d_sampd_dim * layer2_size])

            layer3 = tf.matmul(flattened, weights3) + biases3       # multiply features by weights vector and add biases

            fc_layer1 = tf.nn.relu(features=layer3)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            ''' 
            Apply dropout before the readout layer to reduce overfitting. 
            keep_prob is the probability that any single unit's output is kept during dropout. 

            With this approach, dropout can be turned on during training 
            and off during testing (by setting it to 1.0).

            Additionally, the tf.nn.dropout operation automatically handles scaling unit outputs, 
            so dropout doesn't need any additional scaling.
            '''
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            fc_layer1_dropped = tf.nn.dropout(fc_layer1, keep_prob)

        # Map the 1024 features to NUM_CLASSES
        with tf.name_scope('fc2'):
            # final output layer
            weights4 = self.weight_variable([fc1_size, self.NUM_CLASSES], name='fc2_weights')
            biases4 = self.bias_variable([self.NUM_CLASSES], name='fc2_biases')

            y_conv = tf.matmul(fc_layer1_dropped, weights4) + biases4

        return y_conv, keep_prob

    def test_tflearn_convnet(self):

        '''
        This is a test of a non-custom convnet implementation using TFLearn wrapper library for Tensorflow,
        to test what accuracy can be achieved on my current images dataset.

        Currently achieves about 90% accuracy, which is acceptable,
        but for this particular classification problem it should be possible to get much higher than that.
        I would guess it can reach at least 95%, because Xenopus tadpoles have a very distinct pattern and shape.
        '''

        N_EPOCHS = 100
        # My training dataset is currently too small to run more epochs,
        # the convnet is probably overfitting already

        self.load_data('train_data.npy', 'test_data.npy', "", save_type="np")

        train = self.train_data[:-(self.VAL_SPLIT*len(self.train_data[0]))]
        val = self.train_data[-(self.VAL_SPLIT*len(self.train_data[0])):]

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
                str_label = 'not tadpole'      # detecting tadpoles is the only classification of interest
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

    def test_model(self, path, model_name, type_to_restore="SavedModel", show_incorrect=True):
        """

        :param path:
        :param model_name:
        :param type_to_restore: String, "SavedModel" or "SavedSession"
        :param show_incorrect:  show incorrectly guessed images
        :return:
        """

        print("\n{}".format("#" * 50))

        # self.test_data = self.create_test_data("dataset/test", "test_data", typ="np")

        test_imgs, test_lbls = self.test_data[0], self.test_data[1]
        test_batch = 5

        length = len(test_lbls)

        n_correct = 0
        total_n = length

        with tf.Session(graph=tf.Graph()) as sess:

            graph = tf.get_default_graph()

            if type_to_restore == "SavedModel":
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], path + model_name)
            else:
                saver = tf.train.import_meta_graph(path + model_name)
                saver.restore(sess, tf.train.latest_checkpoint(path))

            x_img = graph.get_tensor_by_name("x_img:0")
            y_lbl = graph.get_tensor_by_name("y_lbl:0")
            predictor = graph.get_tensor_by_name("accuracy/prediction:0")
            keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")

            i = 0
            lngth = len(test_lbls)

            while i < lngth:
                # ensure that all images in test set are tested, even if final set has size less than `test_batch`
                if (lngth - test_batch) >= i:
                    imgs = test_imgs[i: (i + test_batch)]
                    lbls = test_lbls[i: (i + test_batch)]
                else:
                    imgs = test_imgs[i:lngth]
                    lbls = test_lbls[i:lngth]

                print("{}\nTest batch {}".format("-" * 15, i / test_batch))

                preds = sess.run(predictor, feed_dict={x_img: imgs, keep_prob: 1.0})

                truths = np.asarray([t[1] for t in lbls])               # TODO:  These indices should match

                print("predictions: \t", preds)
                print("truths: \t\t", truths)

                correct = np.asarray([1 if p == t else 0 for p, t in zip(preds, truths)])
                n_correct += sum(correct)

                acc = float(sum(correct)) / len(correct)
                print("accuracy: {:.3f}".format(acc))

                acc_thresh = 0.8
                if show_incorrect and acc < acc_thresh:

                    print("Accuracy less than {}%, displaying incorrect guesses...".format(acc_thresh))

                    for idx in range(len(correct)):
                        if correct[idx]:
                            continue

                        incorrect_img = imgs[idx]
                        guess = preds[idx]

                        # im =
                        plt.imshow(np.squeeze(incorrect_img, axis=2), cmap='gray')

                        plt.title(
                            (TadpoleConvNet.CLASS_ONE, TadpoleConvNet.CLASS_TWO)[guess] + " (false)"
                        )
                        plt.show()

                i += test_batch     # increment i with test_batch step size

            accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
            print("test final accuracy: ", sess.run(accuracy, feed_dict={x_img: test_imgs, y_lbl: test_lbls, keep_prob: 1.0}))

            print("-" * 30)
            print('Final Test Accuracy: {:.6f}'.format(
                float(n_correct) / total_n)
            )
            print("-" * 30)

        return float(n_correct) / total_n

    def main(self, n_epochs, batch_size, graph_loc, log_step=10, force_replace=True):

        # initialize first layer for input
        x_img = tf.placeholder(tf.float32, shape=[None, TadpoleConvNet.IMG_SIZE, TadpoleConvNet.IMG_SIZE, TadpoleConvNet.NUM_CHANNELS], name='x_img')  # images
        y_lbl = tf.placeholder(tf.float32, shape=[None, TadpoleConvNet.NUM_CLASSES], name='y_lbl')  # labels

        conv_net, keep_prob = self.build_network(x_img)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_lbl, logits=conv_net)
            cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")

            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope('optimizer'):
            training_step = tf.train.AdamOptimizer(learning_rate=TadpoleConvNet.LR).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            prediction = tf.argmax(conv_net, axis=1, name="prediction")
            truth = tf.argmax(y_lbl, axis=1)
            is_correct = tf.equal(prediction, truth)
            is_correct = tf.cast(is_correct, tf.float32)
            accuracy = tf.reduce_mean(is_correct, name="accuracy")

        # TODO: save more parts of graph to visualize at end
        print("Saving graph to dir: {}".format(graph_loc))
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(graph_loc)
        train_writer.add_graph(tf.get_default_graph())

        if not force_replace:
            pth_n = 1
            while os.path.exists("./{}{}/".format(self.model_save, pth_n)):
                pth_n += 1
        else:
            pth_n = ''
            # todo: delete each file in the folder that matches each TF file name

        builder = tf.saved_model.builder.SavedModelBuilder('./{}{}/'.format(self.model_save, pth_n))
        saver = tf.train.Saver()

        start = time.time()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            if not TadpoleConvNet.MAKE_BATCHES:

                train_imgs = self.train_data[0]
                train_lbls = self.train_data[1]

                for i in range(n_epochs):

                    # TODO: VALIDATION SET !!!

                    if (i % log_step == 0) or (i == n_epochs-1):       # print training update every `log_step` iterations
                        current_loss = cross_entropy.eval(feed_dict={x_img: train_imgs, y_lbl: train_lbls, keep_prob: 1.0}) #{x_imgs: batch[0], y_lbls: batch[1], keep_prob: 1.0})
                        self.loss_data.append(current_loss)

                        if i < n_epochs - 1:
                            print('step {}, loss {}'.format(i, current_loss))
                        else:
                            print('step {}, loss {}'.format(n_epochs, current_loss))

                    summary, _ = sess.run([merged, training_step], feed_dict={x_img: train_imgs, y_lbl: train_lbls, keep_prob: 0.5})
                    train_writer.add_summary(summary, i)
                    # training_step.run(feed_dict={x_img: train_imgs, y_lbl: train_lbls, keep_prob: 0.5})

            else:

                training_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)  # tuple of img_data, labels
                iterator = training_dataset.make_initializable_iterator()
                next_element = iterator.get_next()
                sess.run(iterator.initializer)

                for i in range(n_epochs):  # get next batch
                    img_batch = []
                    lbl_batch = []

                    for j in range(batch_size):
                        try:
                            pair = sess.run(next_element)
                        except OutOfRangeError as ignored:
                            # reinitialize iterator
                            sess.run(iterator.initializer)
                            break
                        else:
                            img_batch.append(pair[0])
                            lbl_batch.append(pair[1])

                    if (i % log_step == 0) or (i == n_epochs-1):   # print training update every `log_step` iterations
                        current_loss = cross_entropy.eval(feed_dict={x_img: img_batch, y_lbl: lbl_batch, keep_prob: 1.0})
                        self.loss_data.append(current_loss)

                        if i < n_epochs - 1:
                            print('epoch {}, loss {}'.format(i, current_loss))
                        else:   # final step
                            print('epoch {}, loss {}'.format(n_epochs, current_loss))

                    training_step.run(feed_dict={x_img: img_batch, y_lbl: lbl_batch, keep_prob: 0.5})

            print("Training completed, elapsed time {}m {:.3f}s".format(
                int((time.time() - start) / 60), (time.time() - start) % 60
            ))

            saver.save(sess, './{}{}'.format(self.session_save, pth_n))
            print("Session saved to file")

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.TRAINING],
                signature_def_map={
                    "model": tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"x_img": x_img}, outputs={"prediction": prediction})
                },
                assets_collection=None)

            # Add a second MetaGraphDef for inference.
            builder.add_meta_graph(
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    "model": tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs={"x_img": x_img}, outputs={"prediction": prediction}
                    )
                },
            )

            builder.save()
            print("Model saved to file at {}X/".format(self.model_save))


def main(_):

    tadconv = TadpoleConvNet(model_save_name='SavedModels/SavedModel',
                             session_save_name='SavedModels/SavedSession/session')

    tadconv.load_data(train="dataset/train", test="dataset/test",
                      savetrain="dataset/train_data", savetest="dataset/test_data", save_type="np")

    # tf.app.run(main=tadconv.main(iterations=120, batch_size=40, log_step=10))

    tadconv.main(
        n_epochs=150, batch_size=tadconv.BATCH_SIZE, graph_loc="logs/{}/".format(tadconv.MODEL_NAME),
        log_step=10, force_replace=False
    )

    pth_n = 1
    while os.path.exists("./{}{}/".format(tadconv.model_save, pth_n)):
        pth_n += 1
        if pth_n >= 100:
            # delete the first model and save there?
            raise SystemExit("Delete old models")

    if isinstance(pth_n, int):
        pth_n -= 1  # test the most recent save file name that exists

    print("Testing model {}{}".format(tadconv.model_save, pth_n))

    tadconv.test_model(path='SavedModels/', model_name='SavedModel1', type_to_restore="SavedModel", show_incorrect=True)


# except:
#     print("Could not load images")
#     raise SystemExit()

# tadconv.testConvNet()

# model.save("tadpole_model")


if __name__ == '__main__':
    tf.app.run(main=main)#, argv=[sys.argv[0]] + unparsed)
