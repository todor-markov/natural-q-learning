import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import natural_net

slim = tf.contrib.slim

LEARNING_RATE = 0.01

IMAGE_SIZE = 784
LABEL_SIZE = 10

LAYER_SIZES = [128, 32, LABEL_SIZE]

# Natural Neural Network parameters
NATURAL = True
N_s = 100
T = 100


BATCH_SIZE = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

with tf.variable_scope('natural/net', initializer=slim.xavier_initializer()):
    h = slim.fully_connected(x, 128)
    h = natural_net.whitened_fully_connected(h, 32)
    y = natural_net.whitened_fully_connected(h, 10, activation=None)

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
tf.initialize_all_variables().run(session=sess)
for step in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    if step % T == 0:

        print 'step: %s\r' % (step)

        if NATURAL:
            samples, _ = mnist.train.next_batch(N_s)
            sess.run(reparam, feed_dict={x: samples})
    

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels}))

