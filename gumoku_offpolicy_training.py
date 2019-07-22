import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'))
from gumoku_actor import Actor
from gumoku_critic import Critic
from gumoku_ui import GumokuUI
from gumoku_reward import GumokuReward
import glob
import numpy as np
import random
import time
import itertools
import pickle as pkl

N_GRIDS = 15
PLAYERS = {'white': 0.5, 'black': 1}


class OffPolicyTraining(object):

    def __int__(self, file_dir, n_grids=N_GRIDS, continue_train=True, sample_size=10**4, use_ui=True,
                verbose=0, epoch=50,  actor_name='gumoku_actor.h5', critic_name='gumoku_critic.h5', gamma=0.95,
                actor_nfilter=3, actor_kernalsize=5, actor_poolsize=(2, 2), actor_lr=0.001,  # actor_tau=0.35,
                critic_nfilter=3, critic_kernalsize=5, critic_poolsize=(2, 2), critic_lr=0.001):  # critic_tau=0.35

        self.n_grids = n_grids
        self.actor_name, self.critic_name = actor_name, critic_name
        self.actor_nfilter, self.actor_kernalsize, self.actor_poolsize, self.actor_lr = \
            actor_nfilter, actor_kernalsize, actor_poolsize, actor_lr   # self.actor_tau
        self.critic_nfilter, self.critic_kernalsize, self.critic_poolsize, self.critic_lr = \
            critic_nfilter, critic_kernalsize, critic_poolsize, critic_lr  # self.critic_tau

    @staticmethod
    def generate_random_partial_sample(n_grids, n_used_locs, action_loc=False, win_action=False):
        """
        :param n_grids:
        :param n_used_locs: number of n used locations (before the action)
        :param action_loc: if True, action will be in form of [loc_x, loc_y], else it will be 1d array
        :param win_action: if True, generate action that leads to win in next state
        :return: a tuple of single partial samples (state, action, next_state), (action is done by player 'white' (0.5))
                 make sure state is not a win
        """
        ary_len = (n_grids-1)**2
        n_1 = int(n_used_locs / 2)  # round down
        n_2 = n_1  # n_1 is for white, n_2 is for black
        if n_used_locs % 2 == 1:
            n_2 += 1
        ary = np.zeros(ary_len)
        # initialize results -----------------
        state = np.zeros((n_grids-1, n_grids-1))
        state[([0, 0, 0, 0, 0], [1, 2, 3, 4, 5])] = [PLAYERS['white']]*5
        action = None
        # ------------------ re-generate if state is a win ------------------
        if win_action:  # if set next_state to be winning state
            next_state = None
            while GumokuReward.scan_win(state, PLAYERS['white'], consec=5):
                # random select 5 connected pieces in 2d array, use one of it as the last action
                loc = [random.randint(0, n_grids-2), random.randint(0, n_grids-2)]
                hori_ind, vert_ind, diag_ind, revdiag_ind = GumokuReward.get_cross_inds(n_grids-1, loc)
                inds_list = []
                for inl in [hori_ind, vert_ind, diag_ind, revdiag_ind]:
                    if (inl is not None) and (len(inl[0]) >= 5):
                        inds_list.append(inl)
                win_seq_ind = random.sample(inds_list, 1)[0]
                ind_strt = random.randint(0, len(win_seq_ind[0])-5)
                win_seq_ind_ind = [ind_strt+x for x in range(5)]
                act_ind = random.sample(win_seq_ind_ind, 1)[0]

                action = [win_seq_ind[0][act_ind], win_seq_ind[1][act_ind]]  # winning action
                next_state = np.zeros((n_grids-1, n_grids-1))
                next_state[win_seq_ind] = PLAYERS['white']  # winning sequence
                avlb_locs = np.where(next_state == 0)
                avlb_locs_ind = list(range(len(avlb_locs[0])))
                random.shuffle(avlb_locs_ind)
                inds_1 = avlb_locs_ind[: n_1 - 4]
                inds_2 = avlb_locs_ind[-n_2:]
                next_state[(avlb_locs[0][inds_1], avlb_locs[1][inds_1])] = PLAYERS['white']
                next_state[(avlb_locs[0][inds_2], avlb_locs[1][inds_2])] = PLAYERS['black']

                state = next_state.copy()
                state[action[0], action[1]] = 0
                if not action_loc:  # change action from [loc_x, loc_y] to 1d array
                    tmp = np.zeros((n_grids-1, n_grids-1))
                    tmp[action[0], action[1]] = 1
                    action = tmp.flatten()
        else:  # if set next_state to be none-winning state
            next_state = state.copy()
            while GumokuReward.scan_win(state, PLAYERS['white'], consec=5) | \
                    GumokuReward.scan_win(state, PLAYERS['white'], consec=5):
                ary_inds = list(range(ary_len))
                random.shuffle(ary_inds)  # shuffle indices
                state, action, next_state = ary.copy(), ary.copy(), ary.copy()
                inds_1, inds_2, ind_a = ary_inds[:n_1], ary_inds[n_1: n_1+n_2], ary_inds[-1]  # inds_1 is white, inds_2 is black
                for ids, val in zip([inds_1, inds_2], [PLAYERS['white'], PLAYERS['black']]):
                    state[ids] = val
                action[ind_a] = 1
                next_state = state.copy()
                next_state[ind_a] = PLAYERS['white']
                state = np.array(state).reshape(n_grids-1, n_grids-1)
                next_state = np.array(next_state).reshape(n_grids-1, n_grids-1)
                if action_loc:  # change action from 1d array to [loc_x, loc_y]
                    loc_x, loc_y = np.where(np.array(action).reshape(n_grids-1, n_grids-1) == 1)
                    action = [loc_x[0], loc_y[0]]

        return state, action, next_state

    @staticmethod
    def check_sample(state, action, next_state, action_loc=False, win_action=False):
        """
        :param state:
        :param action:
        :param next_state:
        :param action_loc: if True, input action is in form of [loc_x, loc_y], else it will be 1d array
        :param win_action: if the action leads to wining of next_state
        :return: list of boolean
        """
        correct = []
        if action_loc:
            action_check = np.zeros((state.shape[0], state.shape[1]))
            action_check[action[0], action[1]] = PLAYERS['white']
        else:
            action_check = action.reshape((state.shape[0], state.shape[1]))
            action_check[np.where(action_check == 1)] = PLAYERS['white']
        #print("check if state is correct")
        correct.append(state.shape[0] == state.shape[1])
        correct.append(state.shape == next_state)
        correct.append(set(next_state.flatten()) == {0.0, 0.5, 1.0})
        correct.append(GumokuReward.scan_win(state, PLAYERS['white'], consec=5) is False)
        #print("check if action leads to next_state")
        correct.append(set((next_state - state == action_check).flatten()) == {True})
        #print("check if number of pieces are balanced")
        correct.append(abs(len(np.where(state == PLAYERS['white'])[0]) -
                           len(np.where(state == PLAYERS['black'])[0])) <= 1)
        correct.append(abs(len(np.where(next_state == PLAYERS['white'])[0]) -
                           len(np.where(next_state == PLAYERS['black'])[0])) <= 1)
        if win_action:
            #print("check if next_state is wining state")
            correct.append(GumokuReward.scan_win(next_state, PLAYERS['white'], consec=5))
        else:
            correct.append(GumokuReward.scan_win(next_state, PLAYERS['white'], consec=5) is False)
        return correct

    @staticmethod
    def generate_random_complete_sample(n_grids, n_used_locs, action_loc=False, output_reward=True, win_action=False):
        """
        :param n_grids:
        :param n_used_locs: number of n used locations (before the action)
        :param action_loc: if True, action will be in form of [loc_x, loc_y], else it will be 1d array
        :param output_reward: if True, return reward as well
        :param win_action: if True, generate action that leads to win
        :return: a tuple of single partial samples (state, action, reward, next_state), (action is done by player 'white' (0.5))
        """
        state, action, next_state = OffPolicyTraining.generate_random_partial_sample(
            n_grids, n_used_locs, action_loc, win_action)
        if output_reward:
            reward = GumokuReward.cal_reward(next_state, PLAYERS['white'], winrwd=10, defensive=2)
        else:
            reward = None
        return state, action, reward, next_state




    @staticmethod
    def generate_complete_samples(n_samples, n_grids, n_used_locs, action_loc=False):
        """
        :param n_samples: number of samples to generate
        :param n_grids:
        :param n_used_locs: number of n used locations (before the action)
        :param action_loc: if True, action will be in for of [loc_x, loc_y], else it will be 1d array
        :return: a list of lists of complete samples [state, action, reward, next_state, done]
        """

        #GumokuReward.cal_reward(next_state, _id, winrwd=10, defensive=self.defensive)
        pass





    def select_samples(self):
        pass

    def save_samples(self):
        pass

    def save_models(self, save_critic=True, save_actor=True):
        pass

    def train_critic(self):
        pass

    def predict(self):
        pass

