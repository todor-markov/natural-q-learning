# Original implementation by Oleg Medvedev
# Link to Github gist page for the original implementation:
# https://gist.github.com/omdv/98351da37283c8b6161672d6d555cde6#file-readme-md
# 

import gym
import re
import tensorflow as tf
import numpy as np
import shutil
import sys

from natural_net import NaturalNet

class ExperienceQModel(object):
    def __init__(self, env, log_dir, monitor_file=None, max_memory=10000, discount=.9, n_episodes=300, 
                 n_steps=200, batch_size=100, learning_rate = 0.01, dropout_keep_prob = 1.0,
                 exploration=lambda x: 0.1, stop_training=10):
        
        # Memory replay parameters
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

        # episode scores
        self.game_scores = list()
        self.game_score = 0.

        # exploration
        self.eps = exploration # epsilon-greedy as function of epoch
        
        # environment parameters
        self.env = gym.make(env)
        self.monitor_file = monitor_file
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = int(re.findall('\d+',str(self.env.action_space))[0]) # shameless hack to get a dim of actions
        
        # training parameters
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.n_steps = n_steps # must be equal to episode length
        self.batch_size = batch_size
        self.stop_training = stop_training # stop training after stop_training consecutive wins
        self.consec_wins = 0 # number of consecutive wins to stop training
        self.global_step = 0 # global step

        # Neural Network Parameters
        self.n_hidden_1 = self.n_states
        self.layer_sizes = [self.n_states, self.n_actions]

        print self.n_states
        print self.n_actions

        # Natural Neural Network Parameters
        self.T = 100
        self.epsilon = 0.1
        self.N_s = 100
        
        # Initialize tensorflow parameters
        self.x = tf.placeholder(tf.float32, [None, self.n_states],name='states')
        self.y = tf.placeholder(tf.float32, [None, self.n_actions],name='qvals')
        self.keep_prob = dropout_keep_prob
        self.dropout = tf.placeholder(tf.float32,name='dropout')
        
        # Tensorboard directory - try to clean if exists
        try:
            shutil.rmtree(log_dir)
        except:
            pass
        self.log_dir = log_dir
        
        # define graph
        self.tf_define_model()

    # update game score
    def update_game_score(self,episode_score):
        self.game_scores.append(episode_score)
        if len(self.game_scores) > 100:
            del self.game_scores[0]
        self.game_score = np.mean(self.game_scores)

    # process reward
    def exp_process_reward(self,ts,reward,endgame):
        if ts <= self.n_steps-1 and endgame == True:
            reward = -1.
        elif ts == self.n_steps-1 and endgame == False:
            reward = 1.
        else:
            reward = 0.
        return reward

    # saving to memory
    def exp_save_to_memory(self, states):
        self.memory.append(states.copy())
        if len(self.memory) > self.max_memory:
          del self.memory[0]

    # getting batch of the memory
    def exp_get_batch(self, batch_size):
        len_memory = len(self.memory)
        n_examples = min(len_memory, batch_size)
        inputs = np.zeros((n_examples, self.n_states))
        targets = np.zeros((n_examples, self.n_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,size=n_examples)):
            #get_memory
            states = self.memory[idx]

            # input
            inputs[i] = states['state_t'].astype(np.float32)

            # targets - not correcting those which are not taken, use prediction
            feed_dict = {self.x: states['state_t'].reshape(1,-1), self.dropout: self.keep_prob}
            targets[i] = self.session.run(self.predictor, feed_dict)
            
            # acted action
            feed_dict = {self.x: states['state_tp1'].reshape(1,-1), self.dropout: self.keep_prob}
            Qsa = np.max(self.session.run(self.predictor, feed_dict))

            # check if endgame and if not use Bellman's equation
            if states['endgame']:
                targets[i,states['action']] = states['reward']
            else:
                targets[i,states['action']] = states['reward'] + self.discount * Qsa
        return inputs, targets
    
    # aux to define a weight variable
    def tf_weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    # aux to define a bias
    def tf_bias_variable(self,shape):
        initial = tf.constant(.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    # aux to attach many summaries
    def tf_variable_summaries(self,var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

    # Aux function to define layers
    def tf_nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('inputs'):
                self.tf_variable_summaries(input_tensor, layer_name + '/input')

            with tf.name_scope('weights'):
                weights = self.tf_weight_variable([input_dim, output_dim])
                self.tf_variable_summaries(weights, layer_name + '/weights')

            with tf.name_scope('biases'):
                biases = self.tf_bias_variable([output_dim])
                self.tf_variable_summaries(biases, layer_name + '/biases')

            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.add(tf.matmul(input_tensor, weights),biases)
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                activations = act(preactivate, 'activation')
                tf.histogram_summary(layer_name + '/activations', activations)

            return activations


    # construct network
    def tf_network(self):
        hidden1 = self.tf_nn_layer(self.x, self.n_hidden_1, self.n_hidden_1, 'layer1', act=tf.nn.relu)

        with tf.name_scope('dropout'):
            tf.scalar_summary('dropout_probability', self.dropout)
            dropped = tf.nn.dropout(hidden1, self.dropout)

        qout = self.tf_nn_layer(dropped, self.n_hidden_1, self.n_actions, 'qvalues', act=tf.identity)
        return qout
    

    # Construct model
    def tf_define_model(self):
        
        # Init session
        self.session = tf.Session()

        # Model scope
        with tf.name_scope('Model'):
            natural_net = NaturalNet(self.layer_sizes, self.epsilon,
                    tf.truncated_normal_initializer(stddev=0.1))
            self.predictor, _ = natural_net.inference(self.x)
            self.reparametrize = natural_net.reparam_op(self.x)
            #self.predictor = self.tf_network()

        # Loss
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.square(self.y - self.predictor))

        # Define optimizer
        with tf.name_scope('SGD'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Prepare summaries
        tf.scalar_summary('loss', self.loss)

        # Summary writer
        self.merged_summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.log_dir + '/train', graph=tf.get_default_graph())

        # Initializing the session
        self.session.run(tf.initialize_all_variables())


    # Train loop
    def tf_train_model(self):
        # start open ai monitor
        if self.monitor_file:
            self.env.monitor.start(self.monitor_file,force=True)

        global_step = 0
        # Training cycle
        for epoch in range(self.n_episodes):

            # restart episode
            state_tp1 = self.env.reset()
            endgame = False
            sum_avg_loss = 0.
            sum_max_qval = 0.
            n_explorations = 0.
            episode_score = 0.
            states = {}

            for t in range(self.n_steps):
                self.env.render()
                state_t1 = np.array(state_tp1)
        
                # epsilon-greedy exploration
                if self.consec_wins < self.stop_training and np.random.rand() <= self.eps(epoch):
                    n_explorations += 1
                    action = self.env.action_space.sample()
                else:
                    feed_dict = {self.x: state_t1.reshape(1,-1), self.dropout: self.keep_prob}
                    qvals = self.session.run(self.predictor, feed_dict)
                    sum_max_qval += np.max(qvals)
                    action = np.argmax(qvals)

                # take a next step
                state_tp1, reward, endgame, info = self.env.step(action)
                # print("{:4d}: {}".format(t,endgame))

                # process reward
                reward = self.exp_process_reward(t,reward,endgame)
                episode_score = episode_score + 1.0

                #store experience
                states['action'] = action
                states['reward'] = float(reward)
                states['endgame'] = endgame
                states['state_t'] = np.array(state_t1)
                states['state_tp1'] = np.array(state_tp1)
                self.exp_save_to_memory(states)

                # Training loop
                if self.game_score < 195:
                    # get experience replay
                    x_batch, y_batch = self.exp_get_batch(self.batch_size)
                    # create feed dictionary
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.dropout: self.keep_prob}
                    # training
                    _, loss, summary = self.session.run([self.train_op, self.loss, self.merged_summary_op],
                        feed_dict=feed_dict)
                    # add summary to the summary_writer
                    self.global_step += x_batch.shape[0]
                    self.summary_writer.add_summary(summary,self.global_step)
                    # avg loss
                    sum_avg_loss += loss

                    if global_step % self.T == 0:
                        x_batch, _ = self.exp_get_batch(self.N_s)
                        print 'reparametrizing neural network'
                        self.session.run(self.reparametrize, feed_dict={self.x: x_batch})

                # Check if lost or not
                if (endgame == True) or (endgame == False and t == self.n_steps-1):
                    self.update_game_score(episode_score)
                    print("{:4d}: score={:8.1f}, loss={:6.2f}, max qval={:6.2f}, exp={:6.2f}, game score={:6.2f}".
                        format(epoch+1,episode_score,sum_avg_loss/t,sum_max_qval/t,n_explorations/t,self.game_score))
                    if (t == self.n_steps-1):
                        self.consec_wins +=1
                        episode_score = 0
                        break
                    else:
                        self.consec_wins = 0
                        episode_score = 0
                        break
                global_step += 1

        # close monitor session
        if self.monitor_file:
            self.env.monitor.close()


if __name__ == "__main__":

    model = ExperienceQModel(
        env='CartPole-v0',\
        monitor_file = 'results/cartpole',\
        log_dir = '/tmp/tf/cartpole-256_1e-3_norm',\
        max_memory=40000,\
        discount=.90,\
        n_episodes=400,\
        n_steps=200,\
        batch_size=128,\
        learning_rate = 1.e-3,\
        dropout_keep_prob = 1.0,\
        exploration = lambda x: (60-x)/100. if x<30 else 0.1,\
        stop_training = 10
    )

    model.tf_train_model()

    NUM_RUNS = 1

    convergence_iterations = []

    for i in range(NUM_RUNS):
        model.tf_train_model()
        convergence_iterations.append(model.time_to_convergence)
    
    converged_iterations = [x for x in convergence_percentage if x != None]
    conv_percentage = sum(converged_iterations) / NUM_RUNS
    conv_it_mean = np.mean(converged_iterations)
    conv_it_std = np.std(converged_iterations)
    print conv_percentage
    print conv_it_mean, conv_it_std