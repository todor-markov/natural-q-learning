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

def whitened_fully_connected(h, output_size, activation=tf.nn.relu):

    with tf.variable_scope('whitened/fully_connected'):
        layer_index = len(tf.get_collection('WHITENED_HIDDEN_STATES')) + 1
        tf.add_to_collection('WHITENED_HIDDEN_STATES', h)

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

        # store params in collection for later reuse
        tf.add_to_collection('WHITENED_PARAMS', [V, U, d, c])

        return h

def whitened_conv2d(h, num_outputs, kernel_size,
        stride=1, padding='SAME', activation=tf.nn.relu):

    with tf.variable_scope('whitened/Conv'):
        layer_index = len(tf.get_collection('WHITENED_HIDDEN_STATES')) + 1
        tf.add_to_collection('WHITENED_HIDDEN_STATES', h)

        input_size = h.get_shape()[-1]
        V = tf.get_variable('V_' + str(layer_index),
                (kernel_size, kernel_size, input_size, num_outputs))
        
        U = tf.get_variable('U_' + str(layer_index - 1),
                (1, 1, input_size, input_size), trainable=False,
                initializer=_conv_identity_init())

        prev_h = h
        # whitening 1x1 conv
        h = tf.nn.conv2d(h, U, [1, 1, 1, 1], padding)

        # normal conv
        h = tf.nn.conv2d(h, V, [1, stride, stride, 1], padding)

        if activation:
            h = activation(h)

        # store params in collection for later reuse in reparametrization
        tf.add_to_collection('WHITENED_PARAMS', [V, U])

        return h

def reparam_op(epsilon=0.1):
    # perform inference on samples to later estimate mu and sigma
    out = []
    hidden_states = tf.get_collection('WHITENED_HIDDEN_STATES')
    with tf.variable_scope('natural/net', reuse=True):
        for i, var_list in enumerate(tf.get_collection('WHITENED_PARAMS')):

            # decompose var list
            V = var_list[0]
            U = var_list[1]
            if len(var_list) > 2:
                d = var_list[2]
                c = var_list[3]

            conv = True if len(V.get_shape()) > 2 else False

            # compute canonical parameters 
            if conv:
                V_t = tf.reshape(V, [-1, int(V.get_shape()[2]), int(V.get_shape()[3])])
                U_t = tf.squeeze(U)

                W = _one_sided_batch_matmul(U_t, V_t)
            else:
                W = tf.matmul(U, V)
                b = d - tf.matmul(tf.expand_dims(c, 0), W)

            # treat spatial dimensions of hidden states as part of the batch
            if conv:
                hidden_states[i] = tf.reshape(hidden_states[i],
                        [-1, int(hidden_states[i].get_shape()[-1])])

            mu = tf.reduce_mean(hidden_states[i], 0)
            # estimate mu and sigma with samples from D
            sigma = tf.reduce_mean(_batch_outer_product(hidden_states[i], hidden_states[i]), 0)
            # update c and U from new mu and sigma
            new_c = mu
            # sigma must be self adjoint as it is composed of matrices of the form u*u'
            eig_vals, eig_vecs = tf.self_adjoint_eig(sigma)
            diagonal = tf.diag(tf.rsqrt(eig_vals + epsilon))
            new_U = tf.matmul(tf.transpose(eig_vecs), diagonal)
            new_U_inverse = tf.matrix_inverse(new_U)

            if conv:
                # transform U
                new_U_t = tf.expand_dims(tf.expand_dims(new_U, 0), 0)

                #c = tf.assign(c, new_c)
                U = tf.assign(U, new_U_t)
            
                # update V
                new_V = _one_sided_batch_matmul(new_U_inverse, W)
                new_V = tf.reshape(new_V, V.get_shape())
            else:
                c = tf.assign(c, new_c)
                U = tf.assign(U, new_U)

                # update V and d
                new_V = tf.matmul(new_U_inverse, W)
                new_d = b + tf.matmul(tf.expand_dims(c, 0), tf.matmul(U, new_V))
                new_d = tf.squeeze(new_d, [0])

                d = tf.assign(d, new_d)

            V = tf.assign(V, new_V)
            
            tensors = [tf.reshape((U), [-1]), tf.reshape((V), [-1])]
            if not conv:
                tensors += [c, d]
            out = [tf.concat(0, out + tensors)]
        return out[0] # only exists to provide op for TF to run (there's probably a nicer way of doing this)

