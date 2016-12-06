import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

slim = tf.contrib.slim

LEARNING_RATE = 0.001

IMAGE_SIZE = 784
LABEL_SIZE = 10

NUM_LAYERS = 3
LAYER_SIZES = [IMAGE_SIZE, 128, 32, LABEL_SIZE]

# Natural Neural Network parameters
N_s = 100
EPSILON = 0.1
T = 100

BATCH_SIZE = 100

assert NUM_LAYERS == len(LAYER_SIZES) - 1

def _batch_outer_product(A, B):
    return tf.batch_matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1))

def _identity_init():
    """Identity matrix initializer"""
    def _identity_initializer(shape, **kwargs):
        out = np.identity(shape[0])
        return out
    return _identity_initializer

# for debugging
def _print_params(sess):
    with tf.variable_scope('natural/net', reuse=True):
        V = tf.get_variable('V_2')
        U = tf.get_variable('U_1')
        d = tf.get_variable('d_2')
        c = tf.get_variable('c_1')
        W = tf.matmul(U, V)
        print 'V', tf.reduce_sum(V).eval(session=sess)
        print 'U', tf.reduce_sum(U).eval(session=sess)
        print 'd', tf.reduce_sum(d).eval(session=sess)
        print 'c', tf.reduce_sum(c).eval(session=sess)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

def inference(x, scope='natural/net'):
    hidden_states = []
    with tf.variable_scope(scope, initializer=slim.xavier_initializer()):
        h = x
        hidden_states.append(h)
        for i in range(1, NUM_LAYERS+1):

            # trainable network params
            V = tf.get_variable('V_' + str(i), (LAYER_SIZES[i - 1], LAYER_SIZES[i]))
            d = tf.get_variable('d_' + str(i), (LAYER_SIZES[i], ))

            # whitening params
            U = tf.get_variable('U_' + str(i - 1), (LAYER_SIZES[i - 1], LAYER_SIZES[i - 1]),
                    initializer=_identity_init(), trainable=False)
            c = tf.get_variable('c_' + str(i - 1), (LAYER_SIZES[i - 1], ),
                    initializer=tf.constant_initializer(), trainable=False)

            # whitened layer
            h = tf.matmul(h - c, tf.matmul(U, V)) + d
            if i < NUM_LAYERS - 1:
                h = tf.nn.relu(h)
            hidden_states.append(h)

    return h, hidden_states

def reparametrize(samples):
    # perform inference on samples to later estimate mu and sigma
    out = []
    with tf.variable_scope('natural', reuse=True):
        _, hidden_states = inference(samples, scope='net')
        with tf.variable_scope('net'):
            for i in range(1, NUM_LAYERS+1):
                # fetch relevant variables
                V = tf.get_variable('V_' + str(i))
                d = tf.get_variable('d_' + str(i))
                U = tf.get_variable('U_' + str(i - 1))
                c = tf.get_variable('c_' + str(i - 1))

                # compute canonical parameters 
                W = tf.matmul(U, V)
                W = tf.Print(W, [tf.reduce_sum(V), i])
                b = d - tf.matmul(tf.expand_dims(c, 0), W)

                # estimate mu and sigma with samples from D
                mu = tf.reduce_mean(hidden_states[i - 1], 0)
                sigma = tf.reduce_mean(_batch_outer_product(hidden_states[i - 1], hidden_states[i - 1]), 0)

                # update c and U from new mu and sigma
                new_c = mu
                # sigma must be self adjoint as it is composed of matrices of the form u*u'
                eig_vals, eig_vecs = tf.self_adjoint_eig(sigma)

                diagonal = tf.diag(tf.rsqrt(eig_vals + EPSILON))
                new_U = tf.matmul(tf.transpose(eig_vecs), diagonal)
                new_U_inverse = tf.matmul(eig_vecs, tf.diag(tf.sqrt(eig_vals + EPSILON)))

                c = tf.assign(c, new_c)
                U = tf.assign(U, new_U)

                # update V and d
                new_V = tf.matmul(new_U_inverse, W)
                new_d = b + tf.matmul(tf.expand_dims(c, 0), tf.matmul(U, new_V))
                new_d = tf.squeeze(new_d, [0])

                V = tf.assign(V, new_V)
                d = tf.assign(d, new_d)

                tensors = [c, d, tf.reshape((U), [-1]), tf.reshape((V), [-1])]
                out = [tf.concat(0, out + [c, d, tf.reshape((U), [-1]), tf.reshape((V), [-1])])]

    return out[0] # only exists to provide op for TF to run (there's probably a nicer way of doing this)

            

y, _ = inference(x)
reparam = reparametrize(x)

for v in tf.trainable_variables():
    print v.name

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# Train
tf.initialize_all_variables().run(session=sess)
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _, loss = sess.run([tf.no_op(), cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    if step % T == 0:

        # debug logging
        print 'Before reparametrize'
        _print_params(sess)

        print 'step: %s\r' % (step)
        samples, _ = mnist.train.next_batch(N_s)
        sess.run(reparam, feed_dict={x: samples})

        # seems like shit is going to infinity
        print 'After reparametrize'
        _print_params(sess)
    

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels}))

