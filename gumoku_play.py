import numpy as np
import itertools
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'))
from gumoku_actor import Actor
from gumoku_ui import GumokuUI
from gumoku_auto_train import GumokuAutoTrain
from gumoku_reward import GumokuReward

N_GRIDS = 15
PLAYERS = {'white': 0.5, 'black': 1}


class PlayGumoku(GumokuUI, object):

    def __init__(self, file_dir, n_grids=N_GRIDS, human_first=True, human_player='black', pause_time=0.2,
                 actor_name='gumoku_actor.h5',
                 actor_nfilter=5, actor_kernalsize=3, actor_poolsize=(2, 2), actor_lr=0.001, actor_tau=0.3):
        super().__init__(n_grids=n_grids, piece_size=15, unit=40, pause_time=pause_time, mode='play',
                         human_player=human_player, human_first=human_first)
        self.resigned = False
        self.winner = None
        self.pause = False
        self.player = human_player
        self.action = None
        self.n_steps = 0
        self.machine_reward = []
        self.human_reward = []
        self.state = np.zeros((N_GRIDS-1, N_GRIDS-1))  # all pssible positions
        self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))
        self.actor_nn = Actor(model_name=actor_name, model_dir=file_dir, n_grids=n_grids,
                              n_filter=actor_nfilter, kernal_size=actor_kernalsize,
                              poolsize=actor_poolsize, lr=actor_lr, tau=actor_tau,
                              continue_train=True)

    def reset_all(self):
        self.resigned = False
        self.winner = None
        self.action = None
        self.n_steps = 0
        self.machine_reward = []
        self.human_reward = []
        try:
            self.plot_players_rwd.clf()  # clear the entire plot  # Todo: this doesn't work
        except:
            pass
        self.state = np.zeros((N_GRIDS-1, N_GRIDS-1))  # all pssible positions
        self.available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))
        self.reset_board()

    def update_state(self):
        next_s = self.state.copy()
        next_s[self.action[0], self.action[1]] = PLAYERS[self.player]
        self.state = next_s

    def turn(self):
        self.player = 'white' if self.player == 'black' else 'black'

    def check_done(self, next_s):
        win = GumokuReward.check_win(next_s, self.action, PLAYERS[self.player], consec=5)
        if win:
            self.winner = self.player
        elif (win is False) & (len(self.available_loc) == 0):
            self.winner = 'DRAW'
        else:
            pass  # self.winner is None

    def start_function(self):
        if self.pause is False:  # game is ongoing
            self.canvas.bind("<Button 1>", self.machine_vs_human)
            self.player_button.config(state='disabled')
            self.playfirst_button.config(state='disabled')
            self.start_button.configure(text='game ongoing')
        else:  # game finished
            self.canvas.unbind("<Button 1>")
            self.player_button.config(state='normal')
            self.playfirst_button.config(state='normal')
            self.reset_all()
            self.start_button.configure(text='click to start')
            self.pause = False

    def machine_go(self):
        if self.winner is None:
            # reverse state for action calculation if necessary ----------------
            if self.player == 'black':
                state_ = GumokuAutoTrain.reverse_id([self.state], single=True)[0]
            else:
                state_ = self.state

            # choose action (use state_) ----------------
            self.action = self.actor_nn.target_pred([state_], model_type='target', loc_out=True)[0]  # update action
            #print('machine ({}): {}'.format(self.player, self.action))
            self.add_piece(self.action, self.player)  # add piece in UI
            self.update_state()   # update status
            next_s = self.state.copy()
            self.available_loc.remove(tuple(self.action))  # update available location set

            # update plots ------------------
            self.machine_reward.append(GumokuReward.cal_reward(next_s, PLAYERS[self.player], winrwd=10, defensive=2))
            self.update_plot_players_rwd(self.human_reward, self.machine_reward)

            # check winner after action ------------------
            self.check_done(next_s)  # this will auto update winner

            if self.winner is None:
                self.turn()  # continue the game
            else:  # either win or draw
                self.label_txt.set("Winner: "+self.winner)
                self.pause = True  # pause first, waiting for the start button to be clicked

    def machine_vs_machine(self):
        while True:
            self.machine_go()
            if self.winner is not None:
                self.label_txt.set("Winner: "+self.winner)
                break

    def machine_vs_human(self, eventorigin):
        global coord_x, coord_y
        if (self.player == self.human_player) and (self.winner is None):
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
                self.available_loc.remove(tuple(self.action))  # update available location set
                self.update_state()  # update status
                next_s = self.state.copy()

                # reset coord_x and coord_y -------------
                coord_x = -1
                coord_y = -1
                # update plots ------------------
                self.human_reward.append(GumokuReward.cal_reward(next_s, PLAYERS[self.player], winrwd=10, defensive=2))
                self.update_plot_players_rwd(self.human_reward, self.machine_reward)

                # check winner after action ------------------
                self.check_done(next_s)

                if self.winner is None:
                    self.turn()  # continue the game
                    self.machine_go()  # if disable this line, need 2 clicks to go to the next step
                else:  # either win or draw
                    self.label_txt.set("Winner: "+self.winner)
                    self.pause = True  # pause first, waiting for the start button to be clicked
        else:
            self.machine_go()


env = PlayGumoku(human_first=True, human_player='white', file_dir=os.path.join(os.getcwd(), 'RL', 'Gumoku_DDPG'))














