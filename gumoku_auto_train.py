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

'''
------------------------------------------- DDPG -------------------------------------------
Randomly initialize critic_eval_net(s, a) and critic_target_net with weight_c
        and randomly initialize actor_eval_net(s) and actor_target_net wth weight_a
Initialize replay_buffer 
for each episode:
    Initialize random process N for action exploration (the exploration noise)
    Receive initial observation state s1
    for each step t:
        Select action a_t = actor_eval_net(s_t) + N_t 
        Excute action a_t and observe reward r_t and new state S_tnext
        Store transition (s_t, s_t, r_t, s_tnext) in replay_buffer
        Sample a random minibatch of n transitions (s_i, a_i, r_i, s_inext) from replay_buffer
        Set y_i = r_i + gamma * critic_target_net(s_inext, actor_target_net(s_inext))
        Update critic_eval_net by minimizing loss: L = (1/N) * sum(y_i - critic_eval_net(s_i, a_i))**2
        Update the actor_eval_net using the sampled policy gradient: (1/N) * sum(gradient(critic_eval_net(s, a))gradient(actor_eval_net(s))) 
        Update the target nns:
            weight_c_target = tau * weight_c + (1-tau) * weight_c_target
            weight_a_target = tau * weight_a + (1-tau) * weight_a_target
'''


class GumokuAutoTrain(object):

    def __init__(self, pause_time, file_dir, plot_step, n_grids=N_GRIDS, continue_train=True, memory_length=1000,
                 sample_size=200, max_episode=1000, max_time=18000, gamma=0.95, weight_transfer_steps=1, defensive=2,
                 verbose=0, use_ui=True, epoch=50, sample_strategy=None, randomness_weight=0.5,
                 actor_name='gumoku_actor.h5', critic_name='gumoku_critic.h5',
                 actor_nfilter=3, actor_kernalsize=5, actor_poolsize=(2, 2), actor_lr=0.001, actor_tau=0.35,
                 critic_nfilter=3, critic_kernalsize=5, critic_poolsize=(2, 2), critic_lr=0.001, critic_tau=0.35):

        self.replay_buffer = {i: [] for i in range((n_grids-1)**2)}  # dictionary whose keys are n steps before game over
        if sample_strategy == 'random':
            self.sample_strategy = 'random'
        else:
            if type(sample_strategy) == dict:
                self.sample_strategy = sample_strategy
            else:
                self.sample_strategy = {5: {5: 0.7, 10: 0.2}, 10: {5: 0.5, 10: 0.4}}
                '''
                for the above example, it means:
                when total episodes < 5, sample 70% from self.replay_buffer where key <= 5 (samples nearer to game over)
                                         sample 20% from self.replay_buffer where 5 < key <= 10
                                         sample 10% from other keys of self.replay_buffer
                when 5 < total episodes <= 10, sample 50% from self.replay_buffer where key <= 5 (samples nearer to game over)
                                               sample 40% from self.replay_buffer where 5 < key <= 10
                                               sample 10% from other keys of self.replay_buffer
                '''

        self.use_ui = use_ui
        if self.use_ui:  # initialize UI
            self.game_ui = GumokuUI(n_grids=n_grids, piece_size=15, unit=40, pause_time=pause_time, mode='auto_train')
        self.randomness_weight = randomness_weight
        self.plot_step = plot_step
        self.n_grids = n_grids
        self.defensive = defensive  # degree of defensiveness
        self.winner = None
        self.n_steps = 0
        self.total_step_cnt = 0
        self.eps_cnt = 0
        self.player = random.sample(['white', 'black'], 1)[0]
        self.state = np.zeros((n_grids-1, n_grids-1))  # the initial state: all possible positions
        self.action = None
        self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))
        self.memory_length = memory_length
        self.sample_size = sample_size
        self.gamma = gamma
        self.weight_transfer_steps = int(weight_transfer_steps) if weight_transfer_steps > 1 else 1
        self.actor_name, self.critic_name = actor_name, critic_name
        self.actor_nfilter, self.actor_kernalsize, self.actor_poolsize, self.actor_lr, self.actor_tau = \
            actor_nfilter, actor_kernalsize, actor_poolsize, actor_lr, actor_tau
        self.critic_nfilter, self.critic_kernalsize, self.critic_poolsize, self.critic_lr, self.critic_tau = \
            critic_nfilter, critic_kernalsize, critic_poolsize, critic_lr, critic_tau
        self.epoch = epoch
        self.max_episode = max_episode
        self.max_time = max_time
        self.train_loss = {'actor': [], 'critic': []}  # this stores the the mean loss of each plot_step
        #self.temp_loss_storage = {'actor': [], 'critic': []}  # this stores all the loss in each plot_step, and reset after each plot update
        self.rwd_records = {'rwd_mean': [], 'rwd_std': [], 'step_cnt': [], 'draw': []}  # training performance records

        self.model_dir = os.path.join(file_dir, "models")
        self.trainrecord_dir = os.path.join(file_dir, "train_records")

        filename_base = 'gumoku_train_result_(auto)_'
        allfilenames = [int(os.path.basename(f).replace(filename_base, '').replace('.pkl', ''))
                        for f in glob.glob(os.path.join(self.trainrecord_dir, filename_base+"*.pkl"))]
        if len(allfilenames) > 0:
            self.filename = filename_base+str(max(allfilenames) + 1)+'.pkl'
        else:
            self.filename = filename_base+'0.pkl'

        self.actor_nn = Actor(model_name=actor_name, model_dir=self.model_dir,
                              n_grids=n_grids, n_filter=actor_nfilter, kernal_size=actor_kernalsize,
                              poolsize=actor_poolsize, lr=actor_lr, tau=actor_tau,
                              continue_train=continue_train)
        self.critic_nn = Critic(model_name=critic_name, model_dir=self.model_dir,
                                n_grids=n_grids, n_filter=critic_nfilter, kernal_size=critic_kernalsize,
                                poolsize=critic_poolsize, lr=critic_lr, tau=critic_tau,
                                continue_train=continue_train, verbose=verbose, epoch=epoch)
        print(self.actor_nn.eval_model.summary())
        print(self.critic_nn.eval_model.summary())

    @staticmethod
    def reverse_id(l_s, single=True):
        """
        :param l_s: a list of states
        :param single: if True, than l_s is a single state, which is a np 2d array
        :return: a list of states with reversed player id
        """
        ids = list(PLAYERS.values())
        id_0, id_1 = ids[0], ids[1]
        id_tmp = random.sample(set(range(5))-set(ids+[0]), 1)[0]  # randomly get a temporary value other than player ids
        if single:
            l_s_tmp = np.stack(l_s)  # change list to np array
        else:
            l_s_tmp = l_s.copy()
        l_s_tmp = np.where(l_s_tmp == id_0, id_tmp, l_s_tmp)  # temporarily change id_0 to id_tmp
        l_s_tmp = np.where(l_s_tmp == id_1, id_tmp, l_s_tmp)  # change id_1 to id_0
        l_s_tmp = np.where(l_s_tmp == id_tmp, id_1, l_s_tmp)  # change id_tmp to id_1
        if single:
            return list(l_s_tmp)  # change np array back to list
        else:
            return l_s_tmp

    @staticmethod
    def split_list_to_input_sample(sample_list):
        """
        :param sample_list: list of samples, each sample is a list: [current_state, action, reward, next_state, done]
        :return: (current_state list, action list, reward list, next_state list, done list)
        """
        s_t, a_t, r_t, s_next, d_t = [], [], [], [], []
        for s in sample_list:
            s_t.append(s[0])  # current state
            a_t.append(s[1])  # action
            r_t.append(s[2])  # reward
            s_next.append(s[3])  # next state
            d_t.append(s[4])  # if done (game over): 0: game ongoing; 1: game over
        return s_t, a_t, r_t, s_next, d_t

    @staticmethod
    def get_all_samples_from_dic(d):
        """
        :param d: a dictionary whose values are all lists of samples
                    (samples are in the form of [current_state, action, reward, next_state, done])
                    example: {0: [[current_state0, action0, reward0, next_state0, done0],
                                  [current_state1, action1, reward1, next_state1, done1]],
                              1: [[current_state2, action2, reward2, next_state2, done2]]}
        :return: a list of all samples
        """
        l = []
        for li in list(d.values()):
            l += li
        return l

    @staticmethod
    def random_sample(l, size):
        """
        :param l: list of samples
        :param size: the desired sized sample
        :return: (current_state list, action list, reward list, next_state list, done list)
        """
        if len(l) >= size:  # if more samples in memory than sample size, random sample of mini-batch
            spl_l = random.sample(l, size)
        else:  # if not enough samples, just use everything we have
            print("not enough sample")
            spl_l = l
        return GumokuAutoTrain.split_list_to_input_sample(spl_l)

    @staticmethod
    def balance_sample(d, sample_strategy_i, size):
        """
        :param d: a dictionary whose values are all lists of samples
                    (samples are in the form of [current_state, action, reward, next_state, done])
                    example: {0: [[current_state0, action0, reward0, next_state0, done0],
                                  [current_state1, action1, reward1, next_state1, done1]],
                              1: [[current_state2, action2, reward2, next_state2, done2]]}
        :param sample_strategy_i: a dictionary tells what key to sample how much:
                    it must have at least one key
                    example: {5: 0.7, 10: 0.2} means
                        sample 70% from self.replay_buffer where key <= 5 (samples nearer to game over)
                        sample 20% from self.replay_buffer where 5 < key <= 10
                        sample 10% from other keys of self.replay_buffer
        :param size: total number of samples required
        :return: (current_state list, action list, reward list, next_state list, done list)
        """
        keys = list(sample_strategy_i.keys())
        keys.sort()
        pct_grp = [sample_strategy_i[i] for i in keys]
        key_grp = [[i for i in range(keys[0]+1)]]
        if len(keys) > 1:
            for i in range(1, len(keys)):
                key_grp.append(list(range(keys[i-1]+1, keys[i]+1)))
        total_pct = sum(list(sample_strategy_i.values()))
        if total_pct < 1:
            key_grp.append([i for i in range(keys[-1]+1, max(d.keys()))])
            pct_grp.append(1-total_pct)

        samples = []
        for kg, pct in zip(key_grp, pct_grp):
            sample_kg = []
            for k in kg:
                sample_kg += d[k]
            if len(sample_kg) > int(size*pct):
                samples += random.sample(sample_kg, int(size*pct))
            else:
                samples += sample_kg
        return GumokuAutoTrain.split_list_to_input_sample(samples)

    def step_balance_sample(self):
        all_keys = list(self.sample_strategy.keys())  # len(self.sample_strategy) should be at least 1
        all_keys.sort()
        if self.eps_cnt <= all_keys[0]:
            result = self.sample_strategy[all_keys[0]]
        elif self.eps_cnt > all_keys[-1]:
            result = None
        else:
            result = self.sample_strategy[all_keys[np.where(np.array(all_keys) / self.eps_cnt >= 1)[0][0]]]
        return result

    @staticmethod
    def maintain_replay_buffer_size(d):
        """
        :param d: a dictionary whose values are all lists of samples
                    (samples are in the form of [current_state, action, reward, next_state, done])
                    example: {0: [[current_state0, action0, reward0, next_state0, done0],
                                  [current_state1, action1, reward1, next_state1, done1]],
                              1: [[current_state2, action2, reward2, next_state2, done2]]}
        :return: reduce 1 sample from d (remove the earliest sample from one of the longest list)
        """
        d1 = d.copy()
        k_l = np.array([[k, len(d[k])] for k in d.keys()])
        key_reduce = k_l[k_l[:, 1] == np.nanmax(k_l[:, 1]), 0]  # may be more than 1
        key_reduce = random.sample(list(key_reduce), 1)[0]
        d1[key_reduce] = d1[key_reduce][1:]
        return d1

    @staticmethod
    def select_loc(action_proba, available_loc):
        """
        :param action_proba: a numpy 2d array with dimension (N_GRIDS-1, N_GRIDS-1)
        :param available_loc: a set of available locations in format of (x_index, y_index)
        :return: the best location in form of [loc_x, loc_y]
        """
        mask_loc = set(itertools.product(*[range(N_GRIDS-1), range(N_GRIDS-1)])) - available_loc
        mask_loc = tuple(np.array(list(mask_loc)).T)
        action_proba[mask_loc] = -np.inf
        max_i = max(action_proba.flatten())
        xlist, ylist = np.where(action_proba >= max_i)
        xy = list(zip(xlist, ylist))
        return list(random.sample(xy, 1)[0])

    def turn(self):
        self.player = 'white' if self.player == 'black' else 'black'

    def save_results(self):
        file = {'result': self.rwd_records,
                'model_loss': self.train_loss,
                'param': {'n_grids': self.n_grids, 'memory_length': self.memory_length,
                          'sample_size': self.sample_size, 'gamma': self.gamma,
                          'weight_transfer_steps': self.weight_transfer_steps, 'defensive': self.defensive,
                          'max_episode': self.max_episode, 'max_time': self.max_time,
                          'actor_nfilter': self.actor_nfilter, 'actor_kernalsize':self.actor_kernalsize,
                          'actor_poolsize': self.actor_poolsize, 'actor_lr': self.actor_lr,
                          'actor_tau': self.actor_tau,
                          'critic_nfilter': self.critic_nfilter, 'critic_kernalsize': self.critic_kernalsize,
                          'critic_poolsize': self.critic_poolsize, 'critic_lr': self.critic_lr,
                          'critic_tau': self.critic_tau, 'epoch': self.epoch}}
        pkl.dump(file, open(os.path.join(self.trainrecord_dir, self.filename), 'wb'))
        self.actor_nn.save(self.model_dir, self.actor_name)
        self.critic_nn.save(self.model_dir, self.critic_name)

    def train_models(self, s_t, a_t, r_t, s_next, d_t):
        a_next = self.actor_nn.target_pred(s_next, model_type='target', loc_out=True)
        r_next = self.critic_nn.target_pred(s_next, a_next, model_type='target')
        r_t_new = [r_t[i] + self.gamma * r_next[i] if d_t[i] == 0 else r_t[i] for i in range(len(r_next))]  # Bellman equation

        # Update critic_eval_net by minimizing loss:
        # L = (1/N) * sum(y_i - critic_eval_net(s_i, a_i))**2 ---------- (this is actually MSE)
        self.critic_nn.train_eval_nn(s_t, a_t, r_t_new)

        # Update the actor_eval_net using the sampled policy gradient:
        # (1/N) * sum(gradient(critic_eval_net(s, a))gradient(actor_eval_net(s)))
        a_for_grad = self.actor_nn.target_pred(s_t, model_type='eval', loc_out=False)
        grads = self.critic_nn.gradients(s_t, a_for_grad)[0]  # shape = (number samples, model output length)
        self.actor_nn.train_eval_with_grads(s_t, grads)  # update eval model

        # ----------- calculate model loss -------------------
        actor_mse = self.actor_nn.cal_mse(s_t, a_t, model_type='eval')
        self.train_loss['actor'].append(actor_mse)
        crtic_mse = self.critic_nn.cal_mse(s_t, a_t, r_t, model_type='eval')
        self.train_loss['critic'].append(crtic_mse)

        # transfer weights Todo: should transfer each step or every several steps?
        if self.total_step_cnt % self.weight_transfer_steps == 0:
            self.critic_nn.transfer_weights()
            self.actor_nn.transfer_weights()

    def train(self):
        self.eps_cnt = 0
        total_time = 0
        start_time = time.time()
        self.total_step_cnt = 0
        train_started = False
        episode_samples = []
        while (self.eps_cnt <= self.max_episode) and (total_time <= self.max_time):
            if self.use_ui:
                self.game_ui.reset_board()  # reset UI
            self.winner = None
            self.player = random.sample(['white', 'black'], 1)[0]  # randomly choose a player to start
            self.state = np.zeros((self.n_grids-1, self.n_grids-1))  # the initial state: all possible positions
            self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))

            all_rwds = []
            self.n_steps = 0

            # ================ START of a episode ================
            while self.winner is None:  # for each t
                # reverse state for action calculation if necessary ----------------
                if self.player == 'black':
                    state_ = GumokuAutoTrain.reverse_id([self.state], single=True)[0]
                else:
                    state_ = self.state

                # choose action (use state_) ----------------
                if self.n_steps == 0:
                    self.action = list(random.sample(self.available_loc, 1)[0])
                elif 0 < self.n_steps < 4:  # before each player placed 2 pieces, add randomness for step selection
                    action_proba = self.actor_nn.eval_model.predict(
                        state_.reshape(1, self.n_grids-1, self.n_grids-1, 1)) + \
                                   np.random.normal(0, 1, (N_GRIDS-1)**2).reshape(1, (N_GRIDS-1)**2) * \
                                   self.randomness_weight
                    action_proba = action_proba.reshape(N_GRIDS-1, N_GRIDS-1)
                    self.action = GumokuAutoTrain.select_loc(action_proba, self.available_loc)
                else:
                    self.action = self.actor_nn.target_pred([state_], model_type='target', loc_out=True)[0]

                if self.use_ui:
                    self.game_ui.add_piece(list(self.action), self.player)  # place piece in UI
                self.n_steps += 1

                # update ----------------
                _id = PLAYERS[self.player]
                current_state = self.state
                next_state = current_state.copy()

                next_state[self.action[0], self.action[1]] = PLAYERS[self.player]
                self.state = next_state  # update state
                self.available_loc.remove(tuple(self.action))

                # check win ------------------
                done = 0  # not done
                win = GumokuReward.check_win(next_state, list(self.action), _id, consec=5)
                if win:
                    self.winner = self.player
                    done = 1  # win: end the game
                elif (win is False) & (len(self.available_loc) == 0):
                    self.winner = 'DRAW'
                    done = 1  # draw: end the game
                else:
                    pass  # self.winner is None

                reward = GumokuReward.cal_reward(next_state, _id, winrwd=10, defensive=self.defensive)
                all_rwds.append(reward)  # Todo: is it the best measurement? Any other better indicator?

                # append record to replay_buffer ---------------------
                if self.player == 'white':
                    episode_samples.append([current_state, self.action, reward, next_state, done])
                else:
                    episode_samples.append([GumokuAutoTrain.reverse_id(current_state, single=True), self.action, reward,
                                            GumokuAutoTrain.reverse_id(next_state, single=True), done])

                # train model ------------
                all_samples = GumokuAutoTrain.get_all_samples_from_dic(self.replay_buffer)
                if len(all_samples) >= self.memory_length:
                    # make sure the memory length is not exceeded
                    self.replay_buffer = GumokuAutoTrain.maintain_replay_buffer_size(self.replay_buffer)  # Todo: not working well
                    # sampling
                    if self.sample_strategy == 'random':
                        s_t_, a_t_, r_t_, s_next_, d_t_ = GumokuAutoTrain.random_sample(all_samples, self.sample_size)
                    else:
                        sample_dict = self.step_balance_sample()
                        if sample_dict is None:
                            s_t_, a_t_, r_t_, s_next_, d_t_ = GumokuAutoTrain.random_sample(
                                all_samples, self.sample_size)
                        else:
                            s_t_, a_t_, r_t_, s_next_, d_t_ = GumokuAutoTrain.balance_sample(
                                self.replay_buffer, sample_dict, self.sample_size)

                    # only train when there is enough samples in memory
                    # print('sample size: ', len(s_t_))  # Todo: the sample size is changing all the time!!!!!!!!!!!!!
                    self.train_models(s_t_, a_t_, r_t_, s_next_, d_t_)
                    train_started = True

                # switch player ---------------
                self.turn()

                # update total step count ------------
                self.total_step_cnt += 1

            # ================ END of a episode ================
            # append samples to replay_buffer ----------------
            for n, spl in enumerate(episode_samples[::-1]):  # append the last sample first
                self.replay_buffer[n].append(spl)
            episode_samples = []  # reset episode_samples after each episode

            # append reward records ----------------
            self.rwd_records['step_cnt'].append(self.n_steps)  # Todo: think about if this is necessary???
            if self.winner == 'DRAW':
                self.rwd_records['rwd_mean'].append(np.nanmean(all_rwds))
                self.rwd_records['rwd_std'].append(np.nanstd(all_rwds))  # Todo: change it to 25, 50, 75 pctile
            else:
                self.rwd_records['rwd_mean'].append(np.mean(all_rwds[:-1]))
                self.rwd_records['rwd_std'].append(np.std(all_rwds[:-1]))

            if self.use_ui:
                if self.eps_cnt % self.plot_step == 0:  # update plots in UI every plot_step episodes
                    self.game_ui.update_plot_rwd(self.rwd_records)
                    if train_started:
                        self.game_ui.update_plot_model_loss(self.train_loss)
            self.eps_cnt += 1
            total_time = time.time() - start_time

        # save results -----------------------
        self.save_results()


'''
# test
gm = GumokuAutoTrain(pause_time=0, file_dir=os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'), epoch=10,
                     memory_length=300, sample_size=30,
                     plot_step=1, max_episode=2000, weight_transfer_steps=1, defensive=2, use_ui=True, verbose=0)
gm.train()
'''
