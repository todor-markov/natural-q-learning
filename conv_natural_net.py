import tensorflow as tf
import numpy as np

import sys

slim = tf.contrib.slim

def _batch_outer_product(A, B):
    return tf.batch_matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1))

def _one_sided_batch_matmul(A, B):
    A = tf.tile(tf.expand_dims(A, 0), [int(B.get_shape()[0]),1,1])
    output = tf.batch_matmul(A, B)
    return output

def _identity_init():
    """Identity matrix initializer"""
    def _identity_initializer(shape, **kwargs):
        out = np.identity(shape[0])
        return out
    return _identity_initializer

def _conv_identity_init():
    """Identity matrix initializer"""
    def _identity_initializer(shape, **kwargs):
        out = np.identity(shape[2])
        out = np.expand_dims(out, 0)
        out = np.expand_dims(out, 0)
        return out
    return _identity_initializer

class NaturalNet():

    def __init__(self, layer_sizes, epsilon, initializer=slim.xavier_initializer()):
        # include hidden and output layer sizes
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.epsilon = epsilon
        self.initializer = initializer

    def whitened_fully_connected(self, h, output_size, layer_index, activation=tf.nn.relu):
        input_size = h.get_shape()[-1]
        V = tf.get_variable('V_' + str(layer_index), (input_size, output_size))
        d = tf.get_variable('d_' + str(layer_index), (output_size, ))

        # whitening params
        U = tf.get_variable('U_' + str(layer_index - 1), (input_size, input_size),
                initializer=_identity_init(), trainable=False)
        c = tf.get_variable('c_' + str(layer_index - 1), (input_size, ),
                initializer=tf.constant_initializer(), trainable=False)

        # whitened layer
        h = tf.matmul(h - c, tf.matmul(U, V)) + d

        if activation:
            h = activation(h)

        return h

    def whitened_conv2d(self, h, num_outputs, kernel_size, layer_index,
            stride=1, padding='SAME', activation=tf.nn.relu):
        input_size = h.get_shape()[-1]
        V = tf.get_variable('V_' + str(layer_index),
                (kernel_size, kernel_size, input_size, num_outputs))
        d = tf.get_variable('d_' + str(layer_index), (num_outputs, ))
        
        U = tf.get_variable('U_' + str(layer_index - 1),
                (1, 1, input_size, input_size), trainable=False,
                initializer=_conv_identity_init())
        c = tf.get_variable('c_' + str(layer_index - 1), (input_size, ), trainable=False,
                initializer=tf.constant_initializer())

        prev_h = h
        # whitening 1x1 conv
        h = tf.nn.conv2d(h - c, U, [1, 1, 1, 1], padding)
        #h = tf.Print(h, [tf.reduce_sum(prev_h - h)])

        # normal conv
        h = tf.nn.conv2d(h, V, [1, stride, stride, 1], padding) # + d

        # multiply U and W and see if its equivalent
        V_t = tf.reshape(V, [-1, int(V.get_shape()[2]), int(V.get_shape()[3])])
        U_t = tf.squeeze(U)
        W = _one_sided_batch_matmul(U_t, V_t)
        print 'W', W.get_shape()
        W = tf.reshape(W, [5,5,int(W.get_shape()[1]), int(W.get_shape()[2])])
        test_h = tf.nn.conv2d(prev_h, W, [1, stride, stride, 1], padding) # + d

        h = tf.Print(h, [tf.reduce_sum(h - test_h)])

        

        if activation:
            h = activation(h)

        return h


    def inference(self, x, scope='natural/net'):

        hidden_states = []
        with tf.variable_scope(scope, initializer=self.initializer):

            V = tf.get_variable('V_' + str(1),
                    (5, 5, 1, self.layer_sizes[0]))
            d = tf.get_variable('d_' + str(1), (self.layer_sizes[0], ))
            h = tf.nn.relu(tf.nn.conv2d(x, V, [1, 1, 1, 1], 'SAME') + d)
            hidden_states.append(h)
            h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')

            for i in range(2, self.num_layers+1):

                #h = self.whitened_fully_connected(h, self.layer_sizes[i - 1],
                        #i, activation)
                h = self.whitened_conv2d(h, self.layer_sizes[i - 1], 5, i)
                hidden_states.append(h)
                h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

            print h.get_shape()
            h = slim.flatten(h)
            print h.get_shape()
            h = self.whitened_fully_connected(h, 1024, self.num_layers+1)
            print h.get_shape()
            h = self.whitened_fully_connected(h, 10, self.num_layers+2, activation=None)
            print h.get_shape()

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

                    # transform variables to appropriate shapes if its a convolutional layer
                    conv = True if len(V.get_shape()) > 2 else False

                    # compute canonical parameters 
                    if conv:
                        print 'V', V.get_shape()
                        print 'd', d.get_shape()
                        print 'U', U.get_shape()
                        print 'c', c.get_shape()
                        V_t = tf.reshape(V, [-1, int(V.get_shape()[2]), int(V.get_shape()[3])])
                        U_t = tf.squeeze(U)
                        print 'V_t', V_t.get_shape()
                        print 'U_t', U_t.get_shape()

                        W = _one_sided_batch_matmul(U_t, V_t)
                        b = d - _one_sided_batch_matmul(tf.expand_dims(c, 0), W)
                    else:
                        W = tf.matmul(U, V)
                        b = d - tf.matmul(tf.expand_dims(c, 0), W)
                    print 'W', W.get_shape()
                    print 'b', b.get_shape()

                    # treat spatial dimensions of hidden states as part of the batch
                    if conv:
                        hidden_states[i - 2] = tf.reshape(hidden_states[i - 2],
                                [-1, int(hidden_states[i - 2].get_shape()[-1])])

                    mu = tf.reduce_mean(hidden_states[i - 2], 0)
                    # estimate mu and sigma with samples from D
                    sigma = tf.reduce_mean(_batch_outer_product(hidden_states[i - 2], hidden_states[i - 2]), 0)
                    print 'sigma', sigma.get_shape()
                    print 'mu', mu.get_shape()

                    # update c and U from new mu and sigma
                    new_c = mu
                    # sigma must be self adjoint as it is composed of matrices of the form u*u'
                    eig_vals, eig_vecs = tf.self_adjoint_eig(sigma)
                    diagonal = tf.diag(tf.rsqrt(eig_vals + self.epsilon))
                    new_U = tf.matmul(tf.transpose(eig_vecs), diagonal)
                    new_U_inverse = tf.matrix_inverse(new_U)

                    if conv:
                        # transform U
                        new_U_t = tf.expand_dims(tf.expand_dims(new_U, 0), 0)

                        c = tf.assign(c, new_c)
                        U = tf.assign(U, new_U_t)
                    
                        # update V and d
                        new_V = _one_sided_batch_matmul(new_U_inverse, W)
                        new_d = b + _one_sided_batch_matmul(tf.expand_dims(c, 0),
                                _one_sided_batch_matmul(new_U, new_V))
                        new_d = tf.squeeze(tf.reduce_mean(new_d, [0]))

                        new_V = tf.reshape(new_V, V.get_shape())
                    else:
                        #c = tf.assign(c, new_c)
                        U = tf.assign(U, new_U)

                        # update V and d
                        new_V = tf.matmul(new_U_inverse, W)
                        new_d = b + tf.matmul(tf.expand_dims(c, 0), tf.matmul(U, new_V))
                        new_d = tf.squeeze(new_d, [0])

                    V = tf.assign(V, new_V)
                    #d = tf.assign(d, new_d)
                    
                    tensors = [c, d, tf.reshape((U), [-1]), tf.reshape((V), [-1])]
                    out = [tf.concat(0, out + [c, d, tf.reshape((U), [-1]), tf.reshape((V), [-1])])]

        return out[0] # only exists to provide op for TF to run (there's probably a nicer way of doing this)
