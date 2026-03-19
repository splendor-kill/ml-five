import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tentacle.config import cfg

np.set_printoptions(precision=3)


def load_rep(file):
    with np.load(file) as rep:
        dat = rep['arr_0']
    m = []
    for g in dat:
        for b in g:
            m.append(np.array(b))
    m = np.array(m)
    return m

def trans(board, BOARD_SIZE=15):
    s, a = board[0], board[1]
    s = s.reshape(BOARD_SIZE, BOARD_SIZE, 3)
    s_raw = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=int)
    s_raw[s[:, :, 2] == 1] = 0
    s_raw[s[:, :, 1] == 1] = 2
    s_raw[s[:, :, 0] == 1] = 1
    a = a.reshape(BOARD_SIZE, BOARD_SIZE)
    s_raw[a == 1] = 4
    return s_raw

def show_states(m, offset):
    gs = gridspec.GridSpec(7, 5)
    for i, g in enumerate(gs):
        ax = plt.subplot(g)
        s_raw = trans(m[i + offset])
        ax.matshow(s_raw)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    import os
    rep_file = os.path.join(cfg.REPLAY_MEMORY_DIR, 'replay-20170810-152827.npz')
    m = load_rep(rep_file)
    offset = 44
    show_states(m, offset)
    plt.plot(m[:200, 2])
    plt.show()
