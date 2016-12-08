import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def _batch_outer_product(A, B):
    return tf.batch_matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1))

def _identity_init():
    """Identity matrix initializer"""
    def _identity_initializer(shape, **kwargs):
        out = np.identity(shape[0])
        return out
    return _identity_initializer

class NaturalNet():

    def __init__(self, layer_sizes, epsilon):
        # include hidden and output layer sizes
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.epsilon = epsilon

    def inference(self, x, scope='natural/net'):

        hidden_states = []
        with tf.variable_scope(scope, initializer=slim.xavier_initializer()):

            V = tf.get_variable('V_1', (x.get_shape()[-1], self.layer_sizes[0]))
            d = tf.get_variable('d_1', (self.layer_sizes[0], ))
            h = tf.nn.relu(tf.matmul(x, V) + d)
            hidden_states.append(h)

            for i in range(2, self.num_layers+1):

                # trainable network params
                V = tf.get_variable('V_' + str(i), (self.layer_sizes[i - 2], self.layer_sizes[i - 1]))
                d = tf.get_variable('d_' + str(i), (self.layer_sizes[i - 1], ))

                # whitening params
                U = tf.get_variable('U_' + str(i - 1), (self.layer_sizes[i - 2], self.layer_sizes[i - 2]),
                        initializer=_identity_init(), trainable=False)
                c = tf.get_variable('c_' + str(i - 1), (self.layer_sizes[i - 2], ),
                        initializer=tf.constant_initializer(), trainable=False)

                # whitened layer
                h = tf.matmul(h - c, tf.matmul(U, V)) + d
                if i < self.num_layers:
                    h = tf.nn.relu(h)
                    hidden_states.append(h)

        return h, hidden_states

    def reparam_op(self, samples):
        # perform inference on samples to later estimate mu and sigma
        out = []
        with tf.variable_scope('natural', reuse=True):
            _, hidden_states = self.inference(samples, scope='net')
            with tf.variable_scope('net'):
                for i in range(2, self.num_layers+1):
                    # fetch relevant variables
                    V = tf.get_variable('V_' + str(i))
                    d = tf.get_variable('d_' + str(i))
                    U = tf.get_variable('U_' + str(i - 1))
                    c = tf.get_variable('c_' + str(i - 1))

                    # compute canonical parameters 
                    W = tf.matmul(U, V)
                    b = d - tf.matmul(tf.expand_dims(c, 0), W)

                    # estimate mu and sigma with samples from D
                    mu = tf.reduce_mean(hidden_states[i - 2], 0)
                    sigma = tf.reduce_mean(_batch_outer_product(hidden_states[i - 2], hidden_states[i - 2]), 0)

                    # update c and U from new mu and sigma
                    new_c = mu
                    # sigma must be self adjoint as it is composed of matrices of the form u*u'
                    eig_vals, eig_vecs = tf.self_adjoint_eig(sigma)
                    diagonal = tf.diag(tf.rsqrt(eig_vals + self.epsilon))
                    new_U = tf.matmul(tf.transpose(eig_vecs), diagonal)
                    new_U_inverse = tf.matrix_inverse(new_U)

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
