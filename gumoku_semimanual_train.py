import numpy as np
import itertools
import pickle as pkl
import random
import glob
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'))
from gumoku_actor import Actor
from gumoku_critic import Critic
from gumoku_ui import GumokuUI
from gumoku_auto_train import GumokuAutoTrain
from gumoku_reward import GumokuReward


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


class GumokuSemiManualTrain(GumokuUI, object):

    def __init__(self, file_dir, n_grids=N_GRIDS, pause_time=0.2,
                 continue_train=True, memory_length=100, sample_size=20, model_save_step=5,
                 gamma=0.95, weight_transfer_steps=1, defensive=2, verbose=0, epoch=50,
                 actor_name='gumoku_actor.h5', critic_name='gumoku_critic.h5',
                 actor_nfilter=3, actor_kernalsize=5, actor_poolsize=(2, 2), actor_lr=0.001, actor_tau=0.35,
                 critic_nfilter=3, critic_kernalsize=5, critic_poolsize=(2, 2), critic_lr=0.001, critic_tau=0.35):

        super().__init__(n_grids=n_grids, piece_size=15, unit=40, pause_time=pause_time, mode='semi_human_train')
        self.winner = None
        self.pause = False
        self.player = 'black'  # initialize
        self.action = None
        self.n_steps = 0  # this will be reset to 0 in the beginning of each episode
        self.total_step_cnt = 0  # this will NOT be reset to 0 in the beginning of each episode
        self.model_save_step = model_save_step  # number of episode before saving model and training results
        self.replay_buffer = []
        self.memory_length = memory_length
        self.sample_size = sample_size
        self.gamma = gamma
        self.weight_transfer_steps = weight_transfer_steps
        self.defensive = defensive
        self.machine_reward = []
        self.episode_rwd = []
        self.human_reward = []
        self.train_loss = []
        self.rwd_records = {'rwd_mean': [], 'rwd_std': [], 'step_cnt': []}  # training performance records

        self.model_dir = os.path.join(file_dir, "models")
        self.trainrecord_dir = os.path.join(file_dir, "train_records")

        filename_base = 'gumoku_train_result_(manual)_'
        allfilenames = [int(os.path.basename(f).replace(filename_base, '').replace('.pkl', ''))
                        for f in glob.glob(os.path.join(self.trainrecord_dir, filename_base+"*.pkl"))]
        if len(allfilenames) > 0:
            self.filename = filename_base+str(max(allfilenames) + 1)+'.pkl'
        else:
            self.filename = filename_base+'0.pkl'

        self.state = np.zeros((N_GRIDS-1, N_GRIDS-1))  # all pssible positions
        self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))

        self.actor_name, self.critic_name = actor_name, critic_name
        self.actor_nfilter, self.actor_kernalsize, self.actor_poolsize, self.actor_lr, self.actor_tau = \
            actor_nfilter, actor_kernalsize, actor_poolsize, actor_lr, actor_tau
        self.critic_nfilter, self.critic_kernalsize, self.critic_poolsize, self.critic_lr, self.critic_tau = \
            critic_nfilter, critic_kernalsize, critic_poolsize, critic_lr, critic_tau
        self.epoch = epoch
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

        self.canvas.bind("<Button 1>", self.human_go)

    def reset_all(self):
        self.winner = None
        self.action = None
        self.n_steps = 0
        self.episode_rwd = []
        self.machine_reward = []
        self.human_reward = []
        self.player = random.sample(['white', 'black'], 1)[0]  # randomly choose a player to start
        self.game_txt.set("Game: ongoing")
        try:
            self.plot_players_rwd.clf()  # clear the entire plot  # Todo: this doesn't work
        except:
            pass
        self.state = np.zeros((N_GRIDS-1, N_GRIDS-1))  # all pssible positions
        self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))
        self.reset_board()

    def turn(self):
        self.player = 'white' if self.player == 'black' else 'black'

    def save_results(self):
        file = {'result': self.rwd_records,
                'model_loss': self.train_loss,
                'param': {'n_grids': self.n_grids, 'memory_length': self.memory_length,
                          'sample_size': self.sample_size, 'gamma': self.gamma,
                          'weight_transfer_steps': self.weight_transfer_steps, 'defensive': self.defensive,
                          'actor_nfilter': self.actor_nfilter, 'actor_kernalsize':self.actor_kernalsize,
                          'actor_poolsize': self.actor_poolsize, 'actor_lr': self.actor_lr,
                          'actor_tau': self.actor_tau,
                          'critic_nfilter': self.critic_nfilter, 'critic_kernalsize': self.critic_kernalsize,
                          'critic_poolsize': self.critic_poolsize, 'critic_lr': self.critic_lr,
                          'critic_tau': self.critic_tau, 'epoch': self.epoch}}
        pkl.dump(file, open(os.path.join(self.trainrecord_dir, self.filename), 'wb'))
        self.actor_nn.save(self.model_dir, self.actor_name)
        self.critic_nn.save(self.model_dir, self.critic_name)
        self.model_txt.set("Model: SAVED")

    def check_done(self, next_s):
        win = GumokuReward.check_win(next_s, self.action, PLAYERS[self.player], consec=5)
        if win:
            self.winner = self.player
        elif (win is False) & (len(self.available_loc) == 0):
            self.winner = 'DRAW'
        else:
            pass  # self.winner is None

    def train_models(self):
        s_t, a_t, r_t, s_next, d_t = GumokuAutoTrain.random_sample(self.replay_buffer, self.sample_size)
        # Todo: use a more balanced way of sampling (sample high reward more to balance with low reward samples)??????????
        a_next = self.actor_nn.target_pred(s_next, model_type='target', loc_out=True)
        r_next = self.critic_nn.target_pred(s_next, a_next)
        r_t_new = [r_t[i] + self.gamma * r_next[i] if d_t[i] == 0 else r_t[i] for i in range(len(r_next))]  # Bellman equation

        # Update critic_eval_net by minimizing loss:
        # L = (1/N) * sum(y_i - critic_eval_net(s_i, a_i))**2 ---------- (this is actually MSE)
        self.critic_nn.train_eval_nn(s_t, a_t, r_t_new)

        # Update the actor_eval_net using the sampled policy gradient:
        # (1/N) * sum(gradient(critic_eval_net(s, a))gradient(actor_eval_net(s)))
        a_for_grad = self.actor_nn.target_pred(s_t, model_type='eval', loc_out=False)
        grads = self.critic_nn.gradients(s_t, a_for_grad)[0]
        self.actor_nn.train_eval_with_grads(s_t, grads)

        # transfer weights Todo: should transfer each step or every several steps?
        if self.total_step_cnt % self.weight_transfer_steps == 0:
            self.critic_nn.transfer_weights()
            self.actor_nn.transfer_weights()
        if self.total_step_cnt % self.model_save_step == 0:
            self.save_results()
        else:
            self.model_txt.set("Model: unsaved")

    def update_records(self):
        # only when game over (win or draw)
        if self.winner == 'DRAW':
            self.game_txt.set("Game: DRAW")
            #print(np.nanmean(all_rwds), np.nanstd(all_rwds), step_cnt)
            self.rwd_records['rwd_mean'].append(np.nanmean(self.episode_rwd))
            self.rwd_records['rwd_std'].append(np.nanstd(self.episode_rwd))
        else:  #elif (self.winner != 'DRAW') & (self.winner is not None):
            self.game_txt.set("Game: winner is "+self.player+" ("+('human' if self.player is 'black' else 'machine')+")")
            #print(np.mean(all_rwds[:-1]), np.std(all_rwds[:-1]), step_cnt)
            self.rwd_records['rwd_mean'].append(np.mean(self.episode_rwd[:-1]))
            self.rwd_records['rwd_std'].append(np.std(self.episode_rwd[:-1]))
        self.rwd_records['step_cnt'].append(self.n_steps)

    def machine_go(self):  # default player is always 'white'
        if self.winner is None:
            # choose action (no need to reverse state, since the default machine player is white) ----------------
            if self.n_steps == 0:
                self.action = list(random.sample(self.available_loc, 1)[0])
            elif 0 < self.n_steps < 4:  # before each player placed 2 pieces, add randomness for step selection
                action_proba = self.actor_nn.eval_model.predict(
                    self.state.reshape(1, self.n_grids-1, self.n_grids-1, 1)) + \
                               np.random.normal(0, 1, (N_GRIDS-1)**2).reshape(1, (N_GRIDS-1)**2)
                action_proba = action_proba.reshape(N_GRIDS-1, N_GRIDS-1)
                self.action = GumokuAutoTrain.select_loc(action_proba, self.available_loc)
            else:
                self.action = self.actor_nn.target_pred([self.state], model_type='target', loc_out=True)[0]

            print('--------------------------')
            print(tuple(self.action) in self.available_loc)
            #print('machine ({}): {}'.format(self.player, self.action))
            self.add_piece(self.action, self.player)  # add piece in UI
            self.n_steps += 1

            # update state -----------------------
            current_state, next_state = self.state.copy(), self.state.copy()
            next_state[self.action[0], self.action[1]] = PLAYERS[self.player]
            self.state = next_state
            try:
                self.available_loc.remove(tuple(self.action))  # update available location set
            except:
                print('!!!!!!!!!! Error (Machine): remove ', tuple(self.action))

            # calculate reward ----------
            reward = GumokuReward.cal_reward(next_state, PLAYERS[self.player], winrwd=10, defensive=2)
            self.episode_rwd.append(reward)
            self.machine_reward.append(reward)

            # update plots ------------------
            self.update_plot_players_rwd(self.human_reward, self.machine_reward)

            # check winner after action ------------------
            self.check_done(next_state)  # this will auto update winner
            done = 0 if self.winner is None else 1

            # append record to replay_buffer -------------
            self.replay_buffer.append([current_state, self.action, reward, next_state, done])
            self.replay_buffer = self.replay_buffer[-self.memory_length:]  # make sure the memory length is not exceeded
            # Todo: use a more balanced way of sampling (sample high reward more to balance with low reward samples)??????????

            if len(self.replay_buffer) > self.sample_size:
                # train model ----------
                self.train_models()
                # update model loss ------------
                self.train_loss.append(np.nanmean(self.critic_nn.train_history.history['mean_squared_error']))
                self.update_plot_model_loss(self.train_loss)

            # update step
            self.total_step_cnt += 1

            # go to the next step ------------
            if self.winner is None:
                self.turn()  # continue the game
            else:  # either win or draw
                self.update_records()
                self.reset_all()

    def human_go(self, eventorigin):
        global coord_x, coord_y
        if (self.player == 'black') and (self.winner is None):  # default human player is black
            click_x = eventorigin.x
            click_y = eventorigin.y
            coord_x = click_x / self.unit - 1
            coord_y = click_y / self.unit - 1
            coord_x = int(coord_x) + 1 if coord_x - int(coord_x) > 0.5 else int(coord_x)
            coord_y = int(coord_y) + 1 if coord_y - int(coord_y) > 0.5 else int(coord_y)
            action = [coord_x, coord_y]

            # check if the action is valid ---------------------
            if (tuple(action) in self.available_loc) & (coord_x >= 0) & (coord_y >= 0):
                self.action = action  # update action
                #print('{}: {}'.format(self.player, self.action), tuple(action) in self.available_loc)
                self.add_piece(self.action, self.player)  # add piece in UI

                # update state -----------------------
                current_state, next_state = self.state.copy(), self.state.copy()
                next_state[self.action[0], self.action[1]] = PLAYERS[self.player]
                self.state = next_state
                try:
                    self.available_loc.remove(tuple(self.action))  # update available location set
                except:
                    print('!!!!!!!!!! Error (Human): remove ', tuple(self.action))
                self.n_steps += 1

                # reset coord_x and coord_y -------------
                coord_x = -1
                coord_y = -1

                # calculate reward ----------
                reward = GumokuReward.cal_reward(next_state, PLAYERS[self.player], winrwd=10, defensive=2)
                self.episode_rwd.append(reward)
                self.human_reward.append(reward)

                # update plots ------------------
                self.update_plot_players_rwd(self.human_reward, self.machine_reward)

                # check winner after action ------------------
                self.check_done(next_state)  # this will auto update winner
                done = 0 if self.winner is None else 1

                # append record to replay_buffer (reverse state, since default human play is black) -------------
                self.replay_buffer.append(
                    [GumokuAutoTrain.reverse_id(current_state, single=True), self.action, reward,
                     GumokuAutoTrain.reverse_id(next_state, single=True), done])
                self.replay_buffer = self.replay_buffer[-self.memory_length:]  # make sure the memory length is not exceeded
                # Todo: use a more balanced way of sampling (sample high reward more to balance with low reward samples)??????????

                if len(self.replay_buffer) > self.sample_size:
                    # train model ----------
                    self.train_models()
                    # update model loss ------------
                    self.train_loss.append(np.nanmean(self.critic_nn.train_history.history['mean_squared_error']))
                    self.update_plot_model_loss(self.train_loss)

                # update step --------
                self.total_step_cnt += 1

                # go to the next step ------------
                if self.winner is None:
                    self.turn()  # continue the game
                    self.machine_go()  # if disable this line, need 2 clicks to go to the next step
                else:  # either win or draw
                    self.update_records()
                    self.reset_all()

        else:
            self.machine_go()


'''
gm = GumokuSemiManualTrain(file_dir=os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'))

'''

