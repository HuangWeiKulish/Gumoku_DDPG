import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, MaxPooling2D, concatenate  # GaussianNoise
from keras.optimizers import Adam
import keras.backend as krbkn
import os


class Critic(object):

    def __init__(self, model_name, model_dir, n_grids, n_filter, kernal_size, poolsize, lr, tau,
                 continue_train=True, verbose=0, epoch=50):
        self.state_dim = (n_grids-1, n_grids-1, 1)
        self.action_dim = n_grids-1
        self.n_filter = n_filter
        self.kernal_size = kernal_size
        self.poolsize = poolsize
        #self.train_history = None
        self.tau, self.lr = tau, lr
        self.eval_model = self.build_nn()
        self.target_model = self.build_nn()
        self.verbose = verbose
        self.epoch = epoch
        self.eval_model.compile(Adam(0.01), loss='mean_squared_error', metrics=['mse'])
        self.target_model.compile(Adam(0.01), loss='mean_squared_error', metrics=['mse'])
        self.action_grads = krbkn.function([self.eval_model.input[0], self.eval_model.input[1]],
                                           krbkn.gradients(self.eval_model.output, [self.eval_model.input[1]]))
        if continue_train:
            try:
                self.load_weights(model_dir, model_name)
            except:
                print('unable to load weights for critic networks')

    def build_nn(self):
        state = Input(self.state_dim)
        action = Input((self.action_dim**2,))

        x1 = Convolution2D(filters=self.n_filter, kernel_size=self.kernal_size, strides=1, padding='same',
                           bias_initializer='zeros', activation='relu')(state)
        #x1 = MaxPooling2D(pool_size=self.poolsize, strides=None, padding='valid')(x1)  # Todo: is this necessary?
        x2 = Dense(10)(action)  # Todo: change it to another cnn???
        #x = GaussianNoise(1.0)(x2)
        x = concatenate([Flatten()(x1), x2])
        x = Dense(20, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer='random_uniform')(x)
        return Model([state, action], out)

    def preprocess(self, l_s, l_a, l_r, a_format='1darray'):
        """
        :param l_s: a list of states
        :param l_a: a list of actions
        :param l_r: a list of rewards
        :param a_format: if '1darray', output actions are 2d array with the length of action_dim,
                            otherwise output 3d array of n [loc_x, loc_y]
        :return: ready inputs into model: array of states, shape=(None, 14, 14, 1),
                                          array of actions, shape=(None, action_dim)
        """
        states = np.expand_dims(np.stack(l_s), 3)

        if (a_format == '1darray') & (np.array(l_a).shape[1] == 2):  # output 2d array: n samples of 1d arrays
            board = np.zeros((self.action_dim, self.action_dim))
            actions = []
            for a in l_a:
                tmp_a = board.copy()
                tmp_a[a[0], a[1]] = 1
                actions.append(tmp_a.flatten())
            actions = np.array(actions)
        elif (a_format != '1darray') & (np.array(l_a).shape[1] == self.action_dim):  # output 3d array: n samples of [loc_x, loc_y]
            board = np.zeros(self.action_dim**2)
            actions = []
            for a, s in zip(l_a, l_s):
                used_locs = np.where(s.flatten() != 0)[0]  # get the used locations on each 1d array
                a[np.where(used_locs)] = -1  # change it to negative value so the used locs will not be selected
                loc = random.sample(list(np.where(a >= a.max())[0]), 1)[0]  # random sample (in case if more than 1 max value)
                tmp_a = board.copy()
                tmp_a[loc] = 1
                tmp_a = tmp_a.reshape(self.action_dim, self.action_dim)
                tmp_ax, tmp_ay = np.where(tmp_a == 1)
                actions.append([tmp_ax[0], tmp_ay[0]])
            actions = np.array(actions)
        else:
            actions = np.array(l_a)

        if l_r is not None:
            rewards = np.array(l_r)
        else:
            rewards = None
        return states, actions, rewards

    def train_eval_nn(self, l_s, l_a, l_r, compile=False):
        """
        :param l_s: a list of states
        :param l_a: a list of actions
        :param l_r: a list of rewards
        """
        if compile:
            self.eval_model.compile(Adam(0.01), loss='mean_squared_error', metrics=['mse'])
        states, actions, rewards = self.preprocess(l_s, l_a, l_r)
        self.eval_model.fit([states, actions], rewards, verbose=self.verbose, epochs=self.epoch)
        #self.train_history = self.eval_model.fit([states, actions], rewards, verbose=self.verbose, epochs=self.epoch)

    def target_pred(self, l_s, l_a, model_type='target'):
        """
        :param l_s: a list of states
        :param l_a: a list of actions
        :param l_r: a list of rewards
        :param model_type: if 'eval', prediction using evaluation model, else use target model
        :return: np 1d array of predictions
        """
        states, actions, _ = self.preprocess(l_s, l_a, None)
        if model_type == 'target':
            pred_reward = self.target_model.predict([states, actions])
        else:
            pred_reward = self.eval_model.predict([states, actions])
        return pred_reward.reshape(len(pred_reward))

    def transfer_weights(self):
        eval_W, target_W = self.eval_model.get_weights(), self.target_model.get_weights()
        for i in range(len(eval_W)):
            target_W[i] = self.tau * eval_W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def cal_mse(self, l_s, l_a, l_r, model_type='eval'):
        """
        :param l_s: list of states
        :param l_a: list of actions
        :param l_r: a list of rewards
        :param model_type: if 'eval', prediction using evaluation model, else use target model
        :return: mse
        """
        r_pred = self.target_pred(l_s, l_a, model_type)
        return ((np.array(l_r) - r_pred)**2).mean()

    def save(self, path, name):
        self.eval_model.save_weights(os.path.join(path, name))

    def load_weights(self, path, name):
        self.eval_model.load_weights(os.path.join(path, name))
        self.target_model.load_weights(os.path.join(path, name))

    def gradients(self, states, a_for_grad):
        """
        :param states: list of state
        :param a_for_grad: a list of actions (locations: [loc_x, loc_y])
        :return:
        """
        states_, a_for_grad, _ = self.preprocess(states, a_for_grad, None)
        return self.action_grads([states_, a_for_grad])
