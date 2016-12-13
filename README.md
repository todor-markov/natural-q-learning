# Natural Q Learning

This repository explores the use of natural gradient methods for deep Q learning.

We also provide a lightweight TensorFlow implementation of the Projected Natural Gradient Descent (PRONG) algorithm from DeepMind's recent paper [Natural Neural Networks](https://papers.nips.cc/paper/5953-natural-neural-networks.pdf).

This technique approximates the natural gradient of a neural network through periodic reparametrizations. It is suggested to use PRONG in conjunction with batch normalization as this results in considerably more stable training.

## Usage
Install [TensorFlow](https://www.tensorflow.org/versions/r0.12/get_started/index.html)

First define your model, using the provided functions for whitened layers: 

```python
	import tensorflow as tf
	import natural_net
	slim = tf.contrib.slim

	inputs = ...
	conv_1 = slim.conv2d(inputs, 32, [5, 5], stride = [2, 2])
	conv_1 = slim.batch_norm(conv_1)
    conv_2 = natural_net.whitened_conv2d(conv_1, 64, 5, stride = 2)
	conv_2 = slim.batch_norm(conv_2)

	flat = slim.flatten(conv_2)
	fc_1 = natural_net.whitened_fully_connected(flat, 1024)
	fc_2 = natural_net.whitened_fully_connected(fc_1, 10, activation = None)
```

During training, run the provided 'reparam_op' every T iterations on a batch of samples (typically 10^2 < T < 10^4)
```python
	# NOTE: must construct reparam op after model is defined
	reparam = natural_net.reparam_op()

	sess = tf.Session()
	for step in range(NUM_STEPS):
		train_batch = get_train_batch()
		loss = sess.run(train_op, feed_dict={inputs: train_batch})
		if step % T == 0:
			samples = get_train_batch()
			sess.run(reparam, feed_dict={x: samples})
```

## Experiments
We currently present experiments on three tasks: MNIST, Cartpole and Gridworld.

We use MNIST to validate our PRONG implementation. Compared to vanilla SGD, our PRONG implementation trains faster and generalizes better, achieving ~1% smaller test error during 10000 epochs training on the MNIST data with a simple fully connected network. We see a similar improvement with a more complex convolutional network. 

Results on Cartpole and Gridworld are very preliminary, but it appears that PRONG hurts convergence in Gridworld task and performs similarly to the normal gradient in the Cartpole task.

We adapt open source deep Q learning implementations for the Cartpole and Gridworld environments from [Oleg Medvedev](https://gist.github.com/omdv/98351da37283c8b6161672d6d555cde6) and [Arthur Juliani](https://github.com/awjuliani/DeepRL-Agents).

