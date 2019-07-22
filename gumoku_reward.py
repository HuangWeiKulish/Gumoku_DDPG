import numpy as np

PLAYERS = {'white': 0.5, 'black': 1}


class GumokuReward:

    @staticmethod
    def yield_chunks(l, sep, lencut=5):
        '''
        :param l: list or np 1d array
        :param sep: splitter
        :param lencut: chunk length threshold
        :return: chunks with length >= lencut on l splitted by sep
        '''
        chk = []
        for x in l:
            if x != sep:
                chk.append(x)
            else:
                if len(chk) >= lencut:
                    yield chk
                    chk = []
                else:
                    chk = []
        if len(chk) >= lencut:  # the last chunk
            yield chk

    @staticmethod
    def get_diag(mtrx, loc):
        """
        :param mtrx: a square shape matrix
        :param loc: the indices of a position
        :return: the diagonal and reverse diagonal which pass through the loc
        """
        dim = mtrx.shape[0]
        mtrx_flp = np.flip(mtrx, 1)
        loc_flp = [loc[0], dim-loc[1]-1]
        if loc[1] >= loc[0]:
            diag = mtrx[:(dim - loc[1] + loc[0]), (loc[1] - loc[0]):].diagonal()
        else:
            diag = mtrx[(loc[0]-loc[1]):, :(loc[1]-loc[0])].diagonal()

        if loc_flp[1] >= loc_flp[0]:
            revdiag = mtrx_flp[:(dim - loc_flp[1] + loc_flp[0]), (loc_flp[1] - loc_flp[0]):].diagonal()
        else:
            revdiag = mtrx_flp[(loc_flp[0]-loc_flp[1]):, :(loc_flp[1]-loc_flp[0])].diagonal()

        return diag, revdiag

    @staticmethod
    def get_cross_inds(dim, loc):
        """
        :param dim: square shaped matrix dimension
        :param loc: the indices of a position
        :return: the horizontal, vertical, diagonal and reverse diagonal indices which pass through the loc,
                    in form of tuple(array(x1, x2, ...), array(y1, y2, ...))
        """
        diag_strt = [loc[0] - loc[1], 0] if loc[1] < loc[0] else [0, loc[1] - loc[0]]
        diag_end = [dim-1, dim + loc[1] - loc[0]-1] if loc[1] < loc[0] else [dim + loc[0] - loc[1]-1, dim-1]
        revdiag_bond = lambda x: dim - 1 - x[0]
        revdiag_strt = [min(loc)-1, dim-1] if revdiag_bond(loc) < loc[1] else [0, sum(loc)]
        revdiag_end = [dim-1, min(loc)-1] if revdiag_bond(loc) < loc[1] else [sum(loc), 0]

        if (loc != [0, dim-1]) & (loc != [dim-1, 0]):
            diag_ind = [[diag_strt[0]+x, diag_strt[1]+y] for x, y in
                        zip(range(0, 1+diag_end[0]-diag_strt[0]), range(0, 1+diag_end[1]-diag_strt[1]))]
            diag_ind = tuple(np.array(diag_ind).T)
        else:
            diag_ind = None
        if (loc != [0, 0]) & (loc != [dim-1, dim-1]):
            revdiag_ind = [[revdiag_strt[0]+x, revdiag_strt[1]-y] for x, y in
                           zip(range(0, 1+revdiag_end[0]-revdiag_strt[0]), range(0, 1+revdiag_end[0]-revdiag_strt[0]))]
            revdiag_ind = tuple(np.array(revdiag_ind).T)
        else:
            revdiag_ind = None
        vert_ind = [[loc[0], y] for y in range(0, dim)]
        vert_ind = tuple(np.array(vert_ind).T)
        hori_ind = [[x, loc[1]] for x in range(0, dim)]
        hori_ind = tuple(np.array(hori_ind).T)
        return hori_ind, vert_ind, diag_ind, revdiag_ind

    @staticmethod
    def check5_seq(l, _id, consec=5):
        """
        :param l: 1d array
        :param _id: id to check
        :param consec: number of consecutive _id to be checked
        :return: True if there are consec number of consecutive _id, else False
        """
        yes = False
        cnt = 0
        for i in range(len(l)):
            if l[i] == _id:
                cnt += 1
            else:
                cnt = 0
            if cnt >= consec:
                yes = True
                break
        return yes

    @staticmethod
    def scan_chunk(chk, thres=2, empty_edge_add=2, winrwd=10, win_n=5):  # Todo: delete
        '''
        :param chk: a chunk: 1d array or list
        :param thres: a threshold
        :param winrwd: the extra score to add if there are at least win_n consecutive pieces
        :param win_n: number of consecutive pieces to win
        :return: score of the chunk to topup
        '''
        consec_chk = list(GumokuReward.yield_chunks(chk, 0, lencut=1))  # get consecutive pieces
        addscore = 0
        if len(consec_chk) > 0:
            max_len = 0
            for c in consec_chk:
                if len(c) >= thres:
                    addscore += len(c)
                if len(c) > max_len:
                    max_len = len(c)
            if (chk[0] == 0) & (len(consec_chk[0]) > thres):
                addscore += empty_edge_add
            if (chk[-1] == 0) & (len(consec_chk[-1]) > thres):
                addscore += empty_edge_add
            if max_len >= win_n:
                addscore += winrwd
        return addscore

    @staticmethod
    def scan_sequence(l1, l2, _id, winrwd=10, defensive=2):
        """
        :param l1: the sequence: np 1d array with 0 (available), and player ids
        :param _id: the player id (refer to PLAYERS)
        :param winrwd: the extra score to add if win / deduct if lose
        :param defensive: the score to be deducted for every 3 connected opponent with empty edge in l1 as well as l2
                        larger value leads to a more defensive play
        :return: score of the sequence for _id (the player)
        """
        # _id = 0.5
        #l1=np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0.5, 0, 1, 1, 1, 0, 0, 0])
        #l2=np.array([0, 0, 1, 1, 1, 0.5, 0, 0, 0, 0.5, 0, 1, 1, 1, 0, 0, 0])

        #l1=np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0.5, 0, 1, 1, 1, 0, 0, 0])
        #l2=np.array([0, 0, 1, 1, 1, 0.5, 1, 1, 0, 0, 0, 0.5, 0, 1, 1, 1, 0, 0, 0])

        op = list(set(PLAYERS.values()) - {_id})[0]  # the opponent
        score = 0
        action_pos = np.where(l2 - l1 > 0)[0][0]

        # Rule 1. get all the chunks with length >= 5 on l1 (separated by _id),
        #           for every chunk: deduct marks according to scan_chunk
        chks_op_1 = list(GumokuReward.yield_chunks(l1, _id, lencut=5))

        tmp = l1.copy()
        tmp[action_pos] = -1
        list(GumokuReward.yield_chunks(tmp, _id, lencut=5))


        for c in chks_op:
            score -= GumokuReward.scan_chunk(c, thres=2, empty_edge_add=2, winrwd=winrwd, win_n=5)

        # Rule 2. get all the chunks with length >= 5 on l(separated by op),
        #           for every chunk, add marks according to scan_chunk
        chks_id = list(GumokuReward.yield_chunks(l, op, lencut=5))
        for c in chks_id:
            score += GumokuReward.scan_chunk(c, thres=2, empty_edge_add=2, winrwd=winrwd, win_n=5)

        return score

    @staticmethod
    def get_n_used_locs(state):
        """
        :param state: np 2d array format (N_GRIDS-1, N_GRIDS-1)
        :return: percent of used locations
        """
        return len(np.where(state != 0)[0]) / (state.shape[0]*state.shape[1])

    @staticmethod
    def cal_reward(state, _id, winrwd=10, defensive=2):  # Todo: force the reward to fixed range?????????
        """
        :param state: np 2d array format (N_GRIDS-1, N_GRIDS-1)
        :param _id: the id of player (refer to PLAYERS table)
        :param winrwd: the extra score to add if win / deduct if lose
        :param defensive: larger value leads to a more defensive play
        :return: the total reward yielded by the action
        """
        reward = 0
        # scan the entire state
        # print('horizontal scan')
        for i in range(state.shape[0]):
            reward += GumokuReward.scan_sequence(state[i, :], _id, winrwd, defensive)
        # print('vertical scan')
        for i in range(state.shape[1]):
            reward += GumokuReward.scan_sequence(state[:, i], _id, winrwd, defensive)
        # print('diagonal and reverse scan')
        # left side
        for i in range(state.shape[0]):
            diag, revdiag = GumokuReward.get_diag(state, [i, 0])
            reward += GumokuReward.scan_sequence(diag, _id, winrwd, defensive)
            reward += GumokuReward.scan_sequence(revdiag, _id, winrwd, defensive)
        # right
        for i in range(state.shape[0]):
            diag, revdiag = GumokuReward.get_diag(state, [i, state.shape[1]-1])
            if i == 0:
                reward += GumokuReward.scan_sequence(diag, _id, winrwd, defensive)
            elif i == state.shape[0]-1:
                reward += GumokuReward.scan_sequence(revdiag, _id, winrwd, defensive)
            else:
                reward += GumokuReward.scan_sequence(diag, _id, winrwd, defensive)
                reward += GumokuReward.scan_sequence(revdiag, _id, winrwd, defensive)

        '''
        factor = GumokuReward.get_n_used_locs(state)
        factor = 1.0/(state.shape[0]*state.shape[1]) if factor == 0.0 else factor  # force it to be positive
        if reward > 0:
            reward = reward / factor  # penalise if too many steps done before game over
        else:  # when reward is negative
            reward = reward * factor
        '''
        return reward  # Todo: force reward to be positive??????????????

    @staticmethod
    def check_win(state, loc, _id, consec=5):
        """
        :param state: 2d array
        :param loc: the location through which vertical, horizontal, diagonal and
                    reverse diagonal sequence will be checked
        :param _id: id to check
        :param consec: number of consecutive _id to be checked
        :return: Check if loc (the position of the action) leads to win
                True if there are consec number of consecutive _id horizonally, or vertically or diangally, else False
        """
        # print('check horizontal')
        win = GumokuReward.check5_seq(state[loc[0], :], _id, consec)
        if win is False:
            # print('check vertical')
            win = GumokuReward.check5_seq(state[:, loc[1]], _id, consec)
            if win is False:
                diag, revdiag = GumokuReward.get_diag(state, loc)
                # print('check diagonal')
                win = GumokuReward.check5_seq(diag, _id, consec)
                if win is False:
                    # print('check reverse diagonal')
                    win = GumokuReward.check5_seq(revdiag, _id, consec)
        return win

    @staticmethod
    def scan_win(state, _id, consec=5):  # Todo: test this function
        """
        :param state: 2d array
        :param _id: id to check
        :param consec: number of consecutive _id to be checked
        :return: Check if state is a win already
                True if there are consec number of consecutive _id horizonally, or vertically or diangally, else False
        """
        win = False
        dim = state.shape[0]
        # print('horizontal scan')
        i = 0
        while not win:
            win = GumokuReward.check5_seq(state[i, :], _id, consec)
            i += 1
            if i >= dim:
                break
        # print('vertical scan')
        i = 0
        while not win:
            win = GumokuReward.check5_seq(state[:, i], _id, consec)
            i += 1
            if i >= dim:
                break
        # print('diagonal and reverse scan')
        # left side
        i = 0
        while not win:
            diag, revdiag = GumokuReward.get_diag(state, [i, 0])
            win = GumokuReward.check5_seq(diag, _id, consec)
            if not win:
                win = GumokuReward.check5_seq(revdiag, _id, consec)
            i += 1
            if i >= dim:
                break
        # right
        i = 0
        while not win:
            diag, revdiag = GumokuReward.get_diag(state, [i, state.shape[1]-1])
            win = GumokuReward.check5_seq(diag, _id, consec)
            if not win:
                win = GumokuReward.check5_seq(revdiag, _id, consec)
            i += 1
            if i >= dim:
                break
        return win






'''
state = np.zeros((14, 14))
#state[1,1], state[2,2], state[3,3], state[4,4] = 1, 1, 1, 1
#state[4,5], state[4,3], state[5,6], state[6,6] = 0.5, 0.5, 0.5, 0.5
# GumokuModel.check_win(next_state, list(self.action), _id, consec=5)
# GumokuModel.cal_reward(current_state, done, next_state, _id, self.action,
#                                                 defensive=1, consec4rwd=10, winrwd=20, min_=0.1, max_=1,
#                                                 base0=0.1, base1=0.6, base2=0.9) 
# test
n_grids = N_GRIDS
state = np.zeros((N_GRIDS-1, N_GRIDS-1))  # the initial state: all possible positions
action = None
available_loc = set(itertools.product(*[range(self.n_grids-1), range(self.n_grids-1)]))
'''
