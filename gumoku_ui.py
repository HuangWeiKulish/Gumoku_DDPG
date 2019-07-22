import numpy as np
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class GumokuUI(tk.Tk, object):

    def __init__(self, n_grids=15, piece_size=15, unit=40, pause_time=0.2,
                 mode='auto_train', human_player='white', human_first=True):
        super(GumokuUI, self).__init__()
        self.n_grids = n_grids
        self.mode = mode
        self.piece_size = piece_size
        self.unit = unit  # both height and width of each grid are 40 pixels
        self.board_dim = n_grids*unit
        self.pause_time = pause_time
        if self.mode == 'play':
            self.human_player = human_player
            self.human_first = human_first
        self.init_board()

    def reset_board(self):
        self.update()
        self.canvas.delete("piece")

    def init_board(self):
        # --------- right part of the ui: game board -------------
        self.canvas = tk.Canvas(self, bg='white', height=self.board_dim, width=self.board_dim)
        # create grids
        for c in range(0, self.board_dim, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.board_dim
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.board_dim, self.unit):
            x0, y0, x1, y1 = 0, r, self.board_dim, r
            self.canvas.create_line(x0, y0, x1, y1)
        self.canvas.grid(row=0, column=0, columnspan=1, rowspan=4, sticky=tk.E+tk.W+tk.S+tk.N)

        # --------- left part of the ui -----------
        if self.mode == 'auto_train':
            # --------- plot reward records -----------------
            self.plot_rwd = Figure(frameon=False, figsize=(6, 2.5))
            self.plot_canvas = FigureCanvasTkAgg(self.plot_rwd, self)
            self.plot_canvas.get_tk_widget().grid(row=0, column=1, columnspan=1, rowspan=1, sticky=tk.W+tk.N)
            # --------- plot model loss -----------------
            self.plot_loss = Figure(frameon=False, figsize=(6, 2.5))
            self.plot_canvas_2 = FigureCanvasTkAgg(self.plot_loss, self)
            self.plot_canvas_2.get_tk_widget().grid(row=1, column=1, columnspan=1, rowspan=1, sticky=tk.W+tk.N)

        elif self.mode == 'play':
            # --------- add label -------------
            self.label_txt = tk.StringVar(self)
            self.label_txt.set("click 'start' button to start the game")
            self.winner_label = tk.Label(self, textvariable=self.label_txt, font=("Helvetica", '16'), fg='purple')
            self.winner_label.grid(row=0, column=1, columnspan=4, rowspan=1, sticky=tk.E+tk.W+tk.N)

            # --------- add buttons --------------
            self.reset_button = tk.Button(self, height=3, width=3, font=('Helvetica', '20'),
                                          text='reset', fg='black', bg='white',  # bg doesn't work
                                          command=self.reset_all)
            self.reset_button.grid(row=1, column=1, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.start_button = tk.Button(self, height=3, width=3, font=('Helvetica', '20'),
                                          text='start', fg='black', bg='white',  # bg doesn't work
                                          command=self.start_function)
            self.start_button.grid(row=1, column=2, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.player_button = tk.Button(self, height=3, width=3, font=('Helvetica', '20'),
                                           text='choose player', fg=self.human_player, bg='blue',  # bg doesn't work
                                           command=self.update_player_button)
            self.player_button.grid(row=1, column=3, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.playfirst_button = tk.Button(self, height=3, width=3, font=('Helvetica', '20'),
                                              text='human first' if self.human_first else 'machine first',
                                              command=self.update_playfirst_button)
            self.playfirst_button.grid(row=1, column=4, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            # --------- plot players' rewards -----------------
            self.plot_players_rwd = Figure(frameon=False, figsize=(6, 3))
            self.plot_canvas_3 = FigureCanvasTkAgg(self.plot_players_rwd, self)
            self.plot_canvas_3.get_tk_widget().grid(row=2, column=1, columnspan=4, rowspan=1,
                                                    sticky=tk.E+tk.W+tk.N+tk.S)

        elif self.mode == 'human_train':
            # --------- add label -------------
            self.totalsteps_txt = tk.StringVar(self)
            self.totalsteps_txt.set("Total steps: 0\nPlayer: white")
            self.totalsteps_label = tk.Label(self, textvariable=self.totalsteps_txt, font=("Helvetica", '16'), fg='blue')
            self.totalsteps_label.grid(row=0, column=1, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.game_txt = tk.StringVar(self)
            self.game_txt.set("Game: ongoing")
            self.game_label = tk.Label(self, textvariable=self.game_txt, font=("Helvetica", '16'), fg='blue')
            self.game_label.grid(row=0, column=2, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.model_txt = tk.StringVar(self)
            self.model_txt.set("Model: unsaved")
            self.model_label = tk.Label(self, textvariable=self.model_txt, font=("Helvetica", '16'), fg='purple')
            self.model_label.grid(row=0, column=3, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            # --------- add button ------------
            self.train_button = tk.Button(self, height=4, width=3, font=('Helvetica', '18'),
                                          text='train\nmodel', fg='black', bg='white',  # bg doesn't work
                                          command=self.train_models)
            self.train_button.grid(row=0, column=4, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            self.savemodel_button = tk.Button(self, height=4, width=3, font=('Helvetica', '18'),
                                              text='save\nmodel', fg='black', bg='white',
                                              command=self.save_results)
            self.savemodel_button.grid(row=0, column=5, columnspan=1, rowspan=1, sticky=tk.E+tk.W+tk.N)

            # --------- plot players' rewards -----------------
            self.plot_players_rwd = Figure(frameon=False, figsize=(6, 2))  # Todo: not working well: legend stick together to the plot
            self.plot_canvas_3 = FigureCanvasTkAgg(self.plot_players_rwd, self)
            self.plot_canvas_3.get_tk_widget().grid(row=1, column=1, columnspan=5, rowspan=1, sticky=tk.W+tk.N+tk.S+tk.E)
            self.rowconfigure(1, weight=3)  # Todo: not working well: legend stick together to the plot

            # --------- plot model loss -----------------
            self.plot_loss = Figure(frameon=False, figsize=(6, 2))  # Todo: not working well: legend stick together to the plot
            self.plot_canvas_2 = FigureCanvasTkAgg(self.plot_loss, self)
            self.plot_canvas_2.get_tk_widget().grid(row=2, column=1, columnspan=5, rowspan=1, sticky=tk.W+tk.N+tk.S+tk.E)
            self.rowconfigure(2, weight=2)  # Todo: not working well: legend stick together to the plot

        else:
            pass

        # --------- pack all ----------
        self.title('Gomoku')
        self.geometry('{0}x{1}'.format(int(self.board_dim*2.05), self.board_dim))

    def add_piece(self, loc, player):
        """
        :param loc: location of the piece: [a, b], a and b are both int, and in range of [0, self.n_grids-1]
        :param player: the color of piece, either 'black' or 'white'
        """
        actual_loc = [self.unit*(loc[0]+1), self.unit*(loc[1]+1)]
        self.piece = self.canvas.create_oval(
            actual_loc[0] - self.piece_size, actual_loc[1] - self.piece_size,
            actual_loc[0] + self.piece_size, actual_loc[1] + self.piece_size,
            fill=player, tags=("piece", ))
        self.update()
        time.sleep(self.pause_time)

    def update_plot_rwd(self, rwd_records):
        rwd_mean = np.array(rwd_records['rwd_mean'])
        rwd_std = np.array(rwd_records['rwd_std'])
        stps = np.array(rwd_records['step_cnt'])
        try:
            self.plot_rwd.clf()  # clear the entire plot
        except:
            pass
        ax1 = self.plot_rwd.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('axes', 1))
        # ----------- plot mean and std reward for each episode -----------
        x = range(1, len(rwd_mean) + 1)
        ax1.plot(x, rwd_mean, c='g', markersize=0.5, marker='', linewidth=0.5, linestyle='-')
        ax1.fill_between(x, rwd_mean-rwd_std, rwd_mean+rwd_std, color='g', alpha=0.1)
        # ----------- plot step count for each episode -----------
        ax2.plot(x, stps, c='b', markersize=0.5,  marker='', linewidth=0.5, linestyle='-')

        ax1.set_ylabel('mean reward', color='g')
        ax2.set_ylabel('step count', color='b')
        ax1.set_title('Reward Plot')
        ax1.set_xlabel('Episode')
        self.plot_rwd.canvas.draw_idle()

    def update_plot_model_loss(self, loss):
        try:
            self.plot_loss.clf()  # clear the entire plot
        except:
            pass
        ax1 = self.plot_loss.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('axes', 1))
        ax1.plot(range(1, len(loss['actor']) + 1), loss['actor'], c='blue',
                 markersize=0.2, marker='', linewidth=0.5, linestyle='-', label='actor')
        ax2.plot(range(1, len(loss['critic']) + 1), loss['critic'], c='orange',
                 markersize=0.2, marker='', linewidth=0.5, linestyle='-', label='critic')
        ax1.set_ylabel('MSE (actor)', color='blue')
        ax2.set_ylabel('MSE (critic)', color='orange')
        ax1.set_xlabel('Episode')
        ax1.set_title('Loss Plot')
        ax1.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        ax2.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        self.plot_loss.canvas.draw_idle()

    def update_plot_players_rwd(self, black_wrd, white_wrd):
        try:
            self.plot_players_rwd.clf()  # clear the entire plot
        except:
            pass
        ax = self.plot_players_rwd.add_subplot(111)
        ax.plot(range(1, len(black_wrd)+1), black_wrd,
                c='black', markersize=0.5, marker='', linewidth=0.5, linestyle='-', label='black reward')
        ax.plot(range(1, len(white_wrd)+1), white_wrd,
                c='orange', markersize=0.5, marker='', linewidth=0.5, linestyle='-', label='white reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Plot')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        self.plot_players_rwd.canvas.draw_idle()

    def update_player_button(self):
        change_to = 'black' if self.human_player == 'white' else 'white'
        self.player_button.configure(fg=change_to, text='player: '+change_to)  # for mac, bg doesn't work, only fg works
        self.human_player = change_to

    def update_playfirst_button(self):
        change_to = False if self.human_first else True
        self.human_first = change_to
        self.playfirst_button.configure(text='human first' if self.human_first else 'machine first')

    def reset_all(self):
        pass

    def start_function(self):
        pass

    def train_models(self):
        pass

    def save_results(self):
        pass



'''
test = GumokuUI()
test.add_piece([2, 3], 'black')
test.reset_board()
test.add_piece([5, 3], 'white')
rwd_records = {'rwd_mean': [3, 4, 5, 4, 5], 'rwd_std': [1, 2, 3, 2, 1], 'step_cnt': [90, 80, 97, 86, 75], 'draw': [0, 0, 0, 1, 0]}
test.update_plot_rwd(rwd_records)
rwd_records = {'rwd_mean': [3, 4, 5, 4, 5, 7, 8, 9, 7], 'rwd_std': [1, 2, 3, 2, 1, 2, 3, 1, 1], 
                'step_cnt': [90, 80, 97, 86, 75, 70, 60, 65, 68], 'draw': [0, 0, 0, 1, 0, 1, 1, 0, 0]}
test.update_plot_rwd(rwd_records)
loss = [4,5,6,7,3,4,2,0]
test.update_plot_model_loss(loss)
'''
'''
test = GumokuUI(mode='play')
test.add_piece([2, 3], 'black')
human_rwd = [4,5,6,7,3,4,2,0]
machine_wrd = [3,4,5,6,4,2,1,0]
test.update_plot_players_rwd(human_rwd, machine_wrd)
test.label_txt.set('reset')
'''

'''
test = GumokuUI(mode='human_train')
test.add_piece([2, 3], 'black')
human_rwd = [4,5,6,7,3,4,2,0]
machine_wrd = [3,4,5,6,4,2,1,0]
test.update_plot_players_rwd(human_rwd, machine_wrd)
model_loss = [3,4,5,6,4,2,1,0]
test.update_plot_model_loss(model_loss)
'''
