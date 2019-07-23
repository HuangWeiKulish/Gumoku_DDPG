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
    def is_sublist(subl, l):
        """
        :param subl: the shorter list
        :param l: the longer list
        :return: boolean: if subl is a sublist of l
        """
        result = False
        sub_len = len(subl)
        for i in range(len(l)):
            if l[i:i+sub_len] == subl:
                result = True
                break
        return result

    @staticmethod
    def chunk_danger(chk, _id):
        """
        :param chk: list or 1d array formed by 0 and / or _id
        :param _id:
        :return: score of danger
                 it's dangerous when:
                 1. at least one 3 consecutive _id with 2 open ends,
                    and chk length > 5:
                        eg: [0, 1, 1, 1, 0, 0] or [1, 0, 0, 1, 1, 1, 0, 0, 1, 1]
                 2. 1 gap between 2 separate consecutive _id, with sum of length >= 3,
                    with 2 open ends, and chk length > 5:
                        eg: [0, 1, 1, 0, 1, 0] or [0, 0, 1, 0, 1, 1, 0]
                 3. at least one 4 consecutive _id, with at least 1 empty edge,
                    and chk length >= 5:
                        eg: [0, 1, 1, 1, 1] or [0, 1, 1, 1, 1, 0]
                 4. 1 gap between 2 separate consecutive _id, with sum of length >= 4, with at least 1 empty edge,
                    and chk length >= 5:
                        eg: [1, 1, 0, 1, 1] or [0, 1, 1, 0, 1, 1]
        """
        danger = 0
        if len(chk) > 5:
            #print('Rule 1')
            if GumokuReward.is_sublist([0, _id, _id, _id, 0, 0], list(chk)) | \
                    GumokuReward.is_sublist([0, 0, _id, _id, _id, 0], list(chk)):
                danger += 1
            #print('Rule 2')
            if GumokuReward.is_sublist([0, _id, _id, 0, _id, 0], list(chk)) | \
                    GumokuReward.is_sublist([0, _id, 0, _id, _id, 0], list(chk)):
                danger += 1
        if len(chk) >= 5:
            #print('Rule 3')
            if GumokuReward.is_sublist([0, _id, _id, _id, _id], list(chk)) | \
                    GumokuReward.is_sublist([_id, _id, _id, _id, 0], list(chk)):
                danger += 1
            #print('Rule 4')
            if GumokuReward.is_sublist([_id, _id, 0, _id, _id], list(chk)) | \
                    GumokuReward.is_sublist([_id, _id, _id, 0, _id], list(chk)) | \
                    GumokuReward.is_sublist([_id, 0, _id, _id, _id], list(chk)):
                danger += 1
        return danger

    @staticmethod
    def scan_sequence(l1, l2, _id, winrwd=10, defensive=1.5):
        """
        :param l1: the sequence: np 1d array with 0 (available), and player ids, the previous sequence
        :param l2: the sequence: np 1d array with 0 (available), and player ids, the next sequence (after action)
        :param _id: the player id (refer to PLAYERS)
        :param winrwd: the extra score to add if win / deduct if lose
        :param defensive: larger value make it more defensive (weight 'not to lose' more than 'win')
        :return: score of the sequence for _id (the player)
        """
        op = list(set(PLAYERS.values()) - {_id})[0]  # the opponent
        score = 0

        # Rule 1: get all the chunks with length >= 5 on l2 (separated by _id),
        #           for every chunk: deduct marks according to danger score
        l1_danger, l2_danger = 0, 0
        chks_op_1 = list(GumokuReward.yield_chunks(l1, _id, lencut=5))  # scan the previous sequence
        for c in chks_op_1:
            l1_danger += GumokuReward.chunk_danger(c, op)
        chks_op_2 = list(GumokuReward.yield_chunks(l2, _id, lencut=5))  # if the next sequence doesn't decrease dangerousity, punish
        for c in chks_op_2:
            l2_danger += GumokuReward.chunk_danger(c, op)

        if l1_danger == 0:
            if l2_danger > 0:
                score -= defensive  # no penalty if l2_danger == 0
        else:  # l1_danger > 0
            if l2_danger > l1_danger:
                score -= 2*defensive
            elif 0 < l2_danger <= l1_danger:
                score -= defensive
            else:  # l2_danger == 0
                score += 2*defensive

        # Rule 2: get all the chunks with length >= 5 on l2 (separated by op),
        #           for every chunk, add marks according to danger score
        l1_favor, l2_favor = 0, 0
        chks_id_1 = list(GumokuReward.yield_chunks(l1, op, lencut=5))
        for c in chks_id_1:
            l1_favor += GumokuReward.chunk_danger(c, _id)
        chks_id_2 = list(GumokuReward.yield_chunks(l2, op, lencut=5))
        for c in chks_id_2:
            l2_favor += GumokuReward.chunk_danger(c, _id)

        if l2_favor > l1_favor:
            if l1_favor == 0:
                score += 1
            else:  # l1_favor > 0
                score += 2

        # Rule 3: check win:
        if GumokuReward.check5_seq(l2, _id, consec=5):
            score += winrwd
        if GumokuReward.check5_seq(l2, op, consec=5):
            score -= winrwd
        return score

    @staticmethod
    def get_n_used_locs(state):
        """
        :param state: np 2d array format (N_GRIDS-1, N_GRIDS-1)
        :return: percent of used locations
        """
        return len(np.where(state != 0)[0]) / (state.shape[0]*state.shape[1])

    @staticmethod
    def cal_reward(state, state_next, _id, winrwd=10, defensive=1.5):  # Todo: force the reward to fixed range?????????
        # Todo: in this way, there will only be reward if at least 3 same pieces have been placed (at least 6 steps)
        """
        :param state: np 2d array format (N_GRIDS-1, N_GRIDS-1)
        :param state_next: np 2d array format (N_GRIDS-1, N_GRIDS-1)
        :param _id: the id of player (refer to PLAYERS table)
        :param winrwd: the extra score to add if win / deduct if lose
        :param defensive: larger value make it more defensive (weight 'not to lose' more than 'win')
        :return: the total reward yielded by the action
        """
        reward = 0
        # scan the entire state
        # print('horizontal scan')
        for i in range(state.shape[0]):
            reward += GumokuReward.scan_sequence(state[i, :], state_next[i, :], _id, winrwd, defensive)
        # print('vertical scan')
        for i in range(state.shape[1]):
            reward += GumokuReward.scan_sequence(state[:, i], state_next[i, :], _id, winrwd, defensive)
        # print('diagonal and reverse scan')
        # left side
        for i in range(state.shape[0]):
            diag, revdiag = GumokuReward.get_diag(state, [i, 0])
            diag_next, revdiag_next = GumokuReward.get_diag(state_next, [i, 0])
            reward += GumokuReward.scan_sequence(diag, diag_next, _id, winrwd, defensive)
            reward += GumokuReward.scan_sequence(revdiag, revdiag_next, _id, winrwd, defensive)
        # right
        for i in range(state.shape[0]):
            diag, revdiag = GumokuReward.get_diag(state, [i, state.shape[1]-1])
            diag_next, revdiag_next = GumokuReward.get_diag(state_next, [i, state_next.shape[1]-1])
            if i == 0:
                reward += GumokuReward.scan_sequence(diag, diag_next, _id, winrwd, defensive)
            elif i == state.shape[0]-1:
                reward += GumokuReward.scan_sequence(revdiag, revdiag_next, _id, winrwd, defensive)
            else:
                reward += GumokuReward.scan_sequence(diag, diag_next, _id, winrwd, defensive)
                reward += GumokuReward.scan_sequence(revdiag, revdiag_next, _id, winrwd, defensive)
        return reward

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
