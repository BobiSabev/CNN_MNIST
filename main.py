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


# TODO experiment with learning rate and batch size
# define parameters
EPOCHS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 50
DROP_OUT = 0.5

TRAIN_SIZE = 2000  # 42000 for full dataset
VALIDATION_SIZE = 500  # train examples to be used for validation: maximum  42000 - TRAIN_SIZE
TEST_SIZE = 100  # 28000 for all test

N_COLLECTED_DATA_POINTS = 1
if N_COLLECTED_DATA_POINTS > 0:
    EVALUATE_EVERY_N_ITER = (EPOCHS * TRAIN_SIZE // BATCH_SIZE) // N_COLLECTED_DATA_POINTS
else:
    EVALUATE_EVERY_N_ITER = (EPOCHS * TRAIN_SIZE // BATCH_SIZE) + 100

IMAGE_SIZE = 28 * 28
N_CLASSES = 10
# TODO Make a real submission - change parameters to
"""
EPOCHS = 25
TRAIN_SIZE = 42000
VALIDATION_SIZE = 0
TEST_SIZE = 28000

And delete model files ?!?
"""


"""
Read data
"""
# init a data object, fill it from
data = Data()
data.read_data(filepath='data/train.csv',
               train_size=TRAIN_SIZE,
               validation_size=VALIDATION_SIZE,
               convert_to_one_hot=True)
# data.train.display_digit()
"""
Now all Data is split into 3 Datasets:
    1. Train data (images, labels) - on this data we train the model at first
    2. Validation set (images, labels) - this data we use to quickly validate the best working model.
After we are done with the model with tuning the model, we can train it on both train and validation
    3. Test set (images) - this is the test set from Kaggle - we have no direct access to the labels. After tuning the model,
an training in on training + validation set, we predict y_test from X_test and write a submission to Kaggle to check result
"""


# TODO try different layer structures
"""
Start Tensorflow session and initialize placeholders
"""
sess = tf.InteractiveSession()
# Placeholder for one image and conversion to 28x28
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
# Placeholder for labels
y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES])
# Placeholder for drop-out probability
keep_prob = tf.placeholder(tf.float32)


"""
Build the neural network layers with TFLearn in network.py
(CONV -> RELU)*2 -> POOL) * 2  -> (FC -> RELU) * 2 -> FC
"""
net = network(input_layer=x, drop_out=DROP_OUT)


"""
Define training parameters
"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y_))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""
Load saved model or train a new one if the there in no model saved
"""
# TODO save the model with the name of the parameters WARNING - might get too much file size and make trouble with git
saver = tf.train.Saver()
path_to_model = "model/model.ckpt"
if os.path.exists("model/checkpoint"):
    print("Loading previously saved models...")
    # Restore variables from disk.
    saver.restore(sess, path_to_model)
    print("Model restored.")
else:
    print("Training a new model")
    sess.run(tf.global_variables_initializer())
# Without the sleep trange makes funny printing
sleep(1)


"""
Check accuracy
"""
# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1


"""
Train the model
"""
print("\nTraining the network...")
t = trange(EPOCHS * data.train.images.shape[0] // BATCH_SIZE)
for i in t:
    # selecting a batch
    batch_x, batch_y = data.train.batch(BATCH_SIZE)
    # evaluating
    if i % EVALUATE_EVERY_N_ITER == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_x,
                                                  y_: batch_y,
                                                  keep_prob: 1.0})
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        t.set_postfix(train_acc=train_accuracy)
        if data.validation.images.shape[0] > 0:
            validation_accuracy = accuracy.eval(feed_dict={x: data.validation.images,
                                                           y_: data.validation.labels,
                                                           keep_prob: 1.0})
            validation_accuracies.append(validation_accuracy)
            t.set_postfix(train_acc=train_accuracy, val_acc=validation_accuracy)
    # Printing evaluation
    t.set_description('iteration %i' % (i + 1))
    # Training
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: DROP_OUT})


"""
VALIDATE
"""
if data.validation.images.shape[0] > 0:
    validation_accuracy = accuracy.eval(feed_dict={x: data.validation.images,
                                                   y_: data.validation.labels,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f' % validation_accuracy)

    if N_COLLECTED_DATA_POINTS >= 2:
        plt.plot(x_range, train_accuracies, '-b', label='Training')
        plt.plot(x_range, validation_accuracies, '-g', label='Validation')
        plt.legend(loc='lower right', frameon=False)
        plt.ylim(ymax=1.1, ymin=0.7)
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.show()


"""
Save the model after training
"""
save_path = saver.save(sess, path_to_model)
print("\nModel saved in file: %s" % save_path)


"""
Validate if any validation data is available
"""
if data.validation.images.shape[0] > 0:
    print("\nTest the network on validation data...")
    print("\ttest accuracy %g" % accuracy.eval(feed_dict={
        x: data.validation.images, y_: data.validation.labels, keep_prob: 1.0}))
else:
    print("\nSkipping validation. No data.")


"""
Make a submission file for Kaggle (Optional)
"""
answer = input("\nDo you want to run on test data and write a Kaggle submission? (y/n)")
if answer == 'y':

    print("Opening test data...")
    test_data_read = p.read_csv('data/test.csv', nrows=TEST_SIZE)
    data.test = Dataset(test_data_read.values, 0)

    print("Generating predictions...")
    prediction = tf.argmax(net, 1)
    data.test.labels = prediction.eval(feed_dict={x: data.test.images, keep_prob: 1.0})

    print("Writing predictions...")
    np.savetxt("data/prediction.csv",
               np.c_[range(1, data.test.labels.shape[0]+1), data.test.labels],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')