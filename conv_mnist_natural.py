import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import sys

import conv_natural_net as natural_net

slim = tf.contrib.slim

LEARNING_RATE = 0.01

IMAGE_SIZE = 784
LABEL_SIZE = 10

# Natural Neural Network parameters
NATURAL = True
N_s = 100
T = 100

CONV = False


BATCH_SIZE = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

x_image = tf.reshape(x, [-1,28,28,1])

#nn = NaturalNet(LAYER_SIZES, EPSILON, slim.xavier_initializer(), conv=True)

with tf.variable_scope('natural/net', initializer=slim.xavier_initializer()):
    h = slim.conv2d(x_image, 32, [5, 5])
    h = slim.max_pool2d(h, [2, 2])

    h = natural_net.whitened_conv2d(h, 64, 5)
    h = slim.max_pool2d(h, [2, 2])

    h = slim.flatten(h)
    h = slim.fully_connected(h, 1024)
    y = slim.fully_connected(h, LABEL_SIZE, activation_fn=None)

reparam = natural_net.reparam_op()

print 'trainable variables'
for v in tf.trainable_variables():
    print v.name

print 'all variables'
for v in tf.all_variables():
    print v.name

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

sess = tf.Session()

tf.scalar_summary('loss', cross_entropy)
merged = tf.merge_all_summaries()
sum_dir = 'summaries/natural' if NATURAL else 'summaries/normal'
summary_writer = tf.train.SummaryWriter(sum_dir, sess.graph)

# Train

accuracies = []

for run in range(1):
    tf.initialize_all_variables().run(session=sess)
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        _, loss, summary = sess.run([train_step, cross_entropy, merged], feed_dict={x: batch_xs, y_: batch_ys})
        summary_writer.add_summary(summary, step)
        if step % T == 0:

            if step % 1 == 0:
                print 'step: %s\r' % (step)

            if NATURAL:
                samples, _ = mnist.train.next_batch(N_s)
                sess.run(reparam, feed_dict={x: samples})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})

    print acc
    accuracies.append(acc)

#print np.mean(np.array(accuracies))

