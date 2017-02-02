import pandas as p
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from time import sleep
import os.path
from data.Data import Data
from data.Dataset import Dataset
from network import network

"""
visualisation
"""

FLAGS = None
# TODO experiment with learning rate and batch size
# define parameters
EPOCHS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 50
DROP_OUT = 0.5
TRAIN_SIZE = 200  # 42000 for full dataset
VALIDATION_SIZE = 500  # train examples to be used for validation: maximum  42000 - TRAIN_SIZE
TEST_SIZE = 100  # 28000 for all test
LOG_DIR = 'MNIST_log'

IMAGE_SIZE = 28 * 28
N_CLASSES = 10  # no hardcoding
# TODO Make a real submission - change parameters to
"""
EPOCHS = 25
TRAIN_SIZE = 42000
VALIDATION_SIZE = 0
TEST_SIZE = 28000

And delete model files ?!?
"""


def train():
    data = Data()
    data.read_data(filepath='data/train.csv',
                   train_size=TRAIN_SIZE,
                   validation_size=VALIDATION_SIZE,
                   convert_to_one_hot=True)

    sess = tf.InteractiveSession()

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    with tf.name_scope('input'):
        input_layer = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
        output_layer = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    with tf.name_scope('reshape_input'):
        image_shaped_input = tf.reshape(input_layer, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def convolution_2d(input_tensor, input_dimension, nb_filter, filter_size, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                weights = weight_variable([filter_size, filter_size, input_dimension, nb_filter])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([nb_filter])
                variable_summaries(biases)
            with tf.name_scope('preactivation'):
                preactivate = conv2d(input_tensor, weights) + biases  # !!!
                tf.summary.histogram('pre-activation', preactivate)
            activations = activation(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def conv2d(input_tensor, weights):
        return tf.nn.conv2d(input_tensor,
                            weights,
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    def max_pool_2d(input_tensor, kernel_size, name):
        with tf.name_scope(name):
            return tf.nn.max_pool(input_tensor,
                                  ksize=[1, 2, 2, 1],  # kernel size?
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

    def fully_connected(input_tensor, image_size, nb_filter, n_units, name, activation):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                weights = weight_variable([image_size*image_size*nb_filter, n_units])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([n_units])
                variable_summaries(biases)
            with tf.name_scope('preactivation'):
                input_tensor_flat = tf.reshape(input_tensor, [-1, image_size*image_size*nb_filter, n_units])
                preactivate = tf.matmul(input_tensor_flat, weights) + biases  # same as convo
                tf.summary.histogram('pre-activation', preactivate)
            if activation == 'NONE':
                return preactivate
            else:
                activations = activation(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

    with tf.name_scope('neural_network_architecture'):
        conv_1 = convolution_2d(image_shaped_input, 1, nb_filter=16, filter_size=3, activation=tf.nn.relu,
                                name='convolutional_layer_1')
        conv_2 = convolution_2d(conv_1, 16, nb_filter=32, filter_size=3, activation=tf.nn.relu,
                                name='convolutional_layer_2')
        pool_1 = max_pool_2d(conv_2, kernel_size=2, name='pool_layer_1')
        conv_3 = convolution_2d(pool_1, 32, nb_filter=64, filter_size=3, activation=tf.nn.relu,
                                name='convolutional_layer_3')
        conv_4 = convolution_2d(conv_3, 64, nb_filter=128, filter_size=3, activation=tf.nn.relu,
                                name='convolutional_layer_4')
        pool_2 = max_pool_2d(conv_4, kernel_size=2, name='pool_layer_2')
        fc_1 = fully_connected(pool_2, 4, nb_filter=128, n_units=2048, activation=tf.nn.relu, name='fully_connected_1')
        fc_2 = fully_connected(fc_1, 1, nb_filter=2048, n_units=512, activation=tf.nn.relu, name='fully_connected_2')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(fc_2, keep_prob)

        y = fully_connected(dropped, 5, nb_filter=512, n_units=10, activation='NONE', name='fully_connected_3')

    with tf.name_scope('loss_function'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=output_layer, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
    tf.global_variables_initializer().run()

    print("\nTraining the network...")
    t = trange(EPOCHS * data.train.images.shape[0] // BATCH_SIZE)
    for i in t:
        # selecting a batch
        batch_x, batch_y = data.train.batch(BATCH_SIZE)
        # evaluating
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={input_layer: data.validation.images,
                                               output_layer: data.validation.labels,
                                               keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={input_layer: data.train.images,
                                                 output_layer: data.train.labels,
                                                 keep_prob: DROP_OUT},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict={input_layer: data.train.images,
                                                                       output_layer: data.train.labels,
                                                                       keep_prob: DROP_OUT})
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    train()

if tf.gfile.Exists(LOG_DIR):
   tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)
train()



"""
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""


