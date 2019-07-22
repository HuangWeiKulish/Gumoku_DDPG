import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, MaxPooling2D  # GaussianNoise,
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as krbkn
import os


class Actor(object):

    def __init__(self, model_dir, model_name, n_grids, n_filter, kernal_size, poolsize, lr, tau,
                 continue_train=True):
        self.state_dim = (n_grids-1, n_grids-1, 1)
        self.action_dim = (n_grids-1)**2
        self.n_filter = n_filter
        self.kernal_size = kernal_size
        self.poolsize = poolsize
        self.tau, self.lr = tau, lr
        self.eval_model = self.build_nn()
        self.target_model = self.build_nn()
        self.eval_model.compile(Adam(0.01), loss='mean_squared_error', metrics=['mse'])
        self.target_model.compile(Adam(0.01), loss='mean_squared_error', metrics=['mse'])
        self.eval_optimizer = self.eval_optimizer()
        if continue_train:
            try:
                self.load_weights(model_dir, model_name)
            except:
                print('unable to load weights for actor networks')

    def build_nn(self):
         state = Input(self.state_dim)
         x = Convolution2D(filters=self.n_filter, kernel_size=self.kernal_size, strides=1, padding='same',
                           bias_initializer='zeros', activation='relu')(state)
         #x = MaxPooling2D(pool_size=self.poolsize, strides=None, padding='valid')(x)  # Todo: is this necessary?
         x = Flatten()(x)
         # x = GaussianNoise(1.0)(x2)
         out = Dense(self.action_dim, activation='relu', kernel_initializer='random_uniform')(x)
         return Model(state, out)

    @staticmethod
    def get_used_locs(s, format_='list_of_locs'):
        """
        :param s: a state, np 2d array, shape = (N_GRIDS-1, N_GRIDS-1)
        :param format_:
        :return: a list of locations that is used up,
                    if format='list_of_locs': return [(loc_x1, loc_y1), (loc_x2, loc_y2), ...]
                    else return return ([loc_x1, loc_x2, ...], [loc_y1, loc_y2, ...])
        """
        if format_ == 'list_of_locs':
            return list(map(tuple, np.array(np.where(s != 0)).T))
        else:
            return tuple(map(list, np.where(s != 0)))

    def convert_action_to_1darray(self, a):
        """
        :param a: example: (loc_x1, loc_y1), or [loc_x1, loc_y1]
        :return: np 1d array with length of (N_GRIDS-1)**2, with value 0 and 1
        """
        emptyboard = np.zeros((self.state_dim[0], self.state_dim[1]))
        emptyboard[a[0], a[1]] = 1
        return emptyboard.flatten()

    def preprocess(self, l_s, l_a, output_state=True, output_action=True, output_used_loc=True, used_loc_format='others'):
        """
        :param l_s: a list of states
        :param l_a: a list of actions
        :param output_state: if True, output pre-processed states
        :param output_action: if True, output pre-processed actions
        :param output_used_loc: if True, output list of available locations
                                    example: [(loc_x1, loc_y1), (loc_x2, loc_y2), ...]
        :param used_loc_format: refer to format in get_used_locs
        :return: ready inputs into model: array of states, shape=(None, 14, 14, 1),
                                          array of actions, shape=(None, (N_GRIDS-1)**2)
                                          array of available locations
        """
        if output_state:
            states = np.expand_dims(np.stack(l_s), 3)
        else:
            states = None
        if output_action:
            actions = np.array([self.convert_action_to_1darray(a) for a in l_a])
        else:
            actions = None
        if output_used_loc:
            used_locs = [Actor.get_used_locs(s, used_loc_format) for s in l_s]
        else:
            used_locs = None
        return states, actions, used_locs

    def cal_mse(self, l_s, l_a, model_type='eval'):
        """
        :param l_s: list of states
        :param l_a: list of actions
        :param model_type: if 'eval', prediction using evaluation model, else use target model
        :return: mse
        """
        actions_pred = self.target_pred(l_s, model_type, loc_out=False)
        _, actions, _ = self.preprocess(l_s, l_a, output_state=False, output_action=True, output_used_loc=False)
        return ((actions_pred - actions)**2).mean()

    def target_pred(self, l_s, model_type, loc_out=True):
        """
        :param l_s: list of states
        :param model_type: if 'eval', prediction using evaluation model, else use target model
        :param loc_out: if True, output a list of the best locations for the next piece,
                                if there are more than one best location, randomly select any
                        else return a list of np 1d array of probabilities (the un-available locations are masked)
                                shape = (196,)
        :return:
        """
        states, _, used_locs = self.preprocess(
            l_s, None, output_state=True, output_action=False, output_used_loc=True, used_loc_format='others')

        if model_type == 'eval':
            y_pred = self.eval_model.predict(states)
        else:
            y_pred = self.target_model.predict(states)
        y_pred = y_pred.reshape(y_pred.shape[0], self.state_dim[0], self.state_dim[1])
        y_pred_modified = []
        for ul, yp in zip(used_locs, y_pred):
            yp[ul] = -1  # change the probability of the used loc to be -1 (the smallest)
            y_pred_modified.append(yp)

        if loc_out:
            max_ = [yi.max() for yi in y_pred_modified]
            xy_list = [list(map(list, np.array(np.where(y_pred_modified[i] >= max_[i])).T))
                       for i in range(len(y_pred_modified))]
            action_pred = [random.sample(xy, 1)[0] for xy in xy_list]
            return action_pred
        else:
            return [yi.flatten() for yi in y_pred_modified]

    def transfer_weights(self):
        eval_W, target_W = self.eval_model.get_weights(), self.target_model.get_weights()
        for i in range(len(eval_W)):
            target_W[i] = self.tau * eval_W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path, name):
        self.eval_model.save_weights(os.path.join(path, name))

    def load_weights(self, path, name):
        self.eval_model.load_weights(os.path.join(path, name))
        self.target_model.load_weights(os.path.join(path, name))

    def eval_optimizer(self):
        action_gdts = krbkn.placeholder(shape=(None, self.action_dim))
        params_grad = tf.gradients(self.eval_model.output, self.eval_model.trainable_weights, -action_gdts)  # Todo: check if this is correct (maximize or minimize???)
        grads = zip(params_grad, self.eval_model.trainable_weights)
        return krbkn.function(inputs=[self.eval_model.input, action_gdts],
                              outputs=[],
                              updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])  # This is to minimize gradient

    def train_eval_with_grads(self, states, grads):
        self.eval_optimizer([states, grads])
