import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
import random


np.random.seed(2)
tf.set_random_seed(2)  # reproducible


GAMMA = 0.95     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

### -----------
class Network:
    def __init__(self,n_actions,input_shape):
        self.input_shape = input_shape
        self.n_actions = n_actions
    def architecture(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)
        Inp = Input(shape = self.input_shape)
        x_cnn = Conv1D(filters=64, kernel_size=3, input_shape=self.input_shape)(Inp)
        #x_cnn = BatchNormalization()(x_cnn)
        x_cnn = Activation('relu')(x_cnn)
        
        x_flatten = Flatten()(x_cnn)
        #x = Dense(20,activation = 'relu')(x_flatten)
        outa  = Dense(self.n_actions,activation = 'softmax')(x_flatten)
        return Model(inputs= [Inp] , outputs = [outa])

    def archi_critic(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)
        Inp = Input(shape = self.input_shape)
        x_cnn = Conv1D(filters=64, kernel_size=3, input_shape=self.input_shape)(Inp)
        #x_cnn = BatchNormalization()(x_cnn)
        x_cnn = Activation('relu')(x_cnn)

        x_flatten = Flatten()(x_cnn)
        outa  = Dense(1,activation = None)(x_flatten)
        return Model(inputs= [Inp] , outputs = [outa])
        

class Actor(object):
    def __init__(self, sess, input_shape, n_actions, lr=0.001,name = 'actor'):
        self.sess = sess
        self.n_actions = n_actions
        self.entropy_factor = 1 # Init value
        self.entropy_decay = 1
        self.learning_step = 0
        #n_features = input_shape[0]
        self.s = tf.placeholder(tf.float32, [None, ] + list(input_shape), "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        with tf.variable_scope(name,reuse  = tf.AUTO_REUSE):
            self.actor_model = Network(n_actions,input_shape).architecture()

        with tf.variable_scope('actor_outputs',reuse = tf.AUTO_REUSE):
            self.acts_prob = self.actor_model(self.s)

        with tf.variable_scope('exp_v',reuse  = tf.AUTO_REUSE):
            self.log_prob = tf.reduce_sum(tf.nn.log_softmax(self.acts_prob) * tf.one_hot(self.a, n_actions),axis = -1)
            self.entropy = - tf.reduce_sum(self.acts_prob*tf.nn.log_softmax(self.acts_prob),axis=-1,name='entropy')
            self.exp_v = -tf.reduce_mean(self.log_prob * self.td_error) - self.entropy_factor * tf.reduce_mean(self.entropy) 
            # advantage (TD_error) guided loss # tf.stop_gradient()

        with tf.variable_scope('train',reuse  = tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)  
            # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        #s = s[np.newaxis, :]
        self.learning_step  = self.learning_step  + 1 
        if(self.learning_step % 100):
            self.entropy_factor = min(0.005,self.entropy_factor - self.entropy_decay)
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
    
            
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        #if np.random.uniform() < 0.1: # add a random noise in 10% of cases
        #    probs = probs + random.uniform(0, 0.2) 
        #    probs = probs/np.sum(probs,axis=-1)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
    
    def predict_action(self,s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.argmax(probs)
        
        
    
    def get_value(self,s,a):
        return self.sess.run(self.log_prob, {self.s: s,self.a : a})

    ################# CRITRIC ###############################
    
class Critic(object):
    def __init__(self, sess, input_shape, lr=0.01,name = 'critic'):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, ] + list(input_shape), "state")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        n_features = input_shape[0]
        with tf.variable_scope(name,reuse  = tf.AUTO_REUSE):
            self.critic_model = Network(1,input_shape).archi_critic()
        

        with tf.variable_scope('critic_outputs',reuse = tf.AUTO_REUSE):
            self.v = self.critic_model(self.s)[:,0]

        with tf.variable_scope('squared_TD_error',reuse = tf.AUTO_REUSE):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train',reuse = tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        #s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
