from random import choice

from pybrain.rl.agents.agent import Agent
from pybrain.rl.environments.twoplayergames.twoplayergame import TwoPlayerGame
from pybrain.structure.networks.custom.convboard import ConvolutionalBoardNetwork
from pybrain.utilities import drawGibbs

import numpy as np
from tentacle.board import Board


class FiveGame(TwoPlayerGame):
    def __init__(self):
        self.reset()
    
    
    def reset(self):
        TwoPlayerGame.reset(self)
        self.movesDone = 0
        self.b = Board()
    
    def isLegal(self, c, pos):
        return self.b.is_legal(pos[0], pos[1])
    
    def _fiveRow(self, c, pos):
        b = self.b.stones.reshape(-1, Board.BOARD_SIZE)
        self.b.find_conn_5(b, pos[0], pos[1], c)
        
    def getLegals(self, c):
        loc = np.where(self.b.stones == 0)
        moves = [i for i in map(lambda i: divmod(i, Board.BOARD_SIZE), loc[0])]
        return moves
    
    def doMove(self, c, pos):
        """ the action is a (color, position) tuple, for the next stone to move.
        returns True if the move was legal. """
        self.movesDone += 1
        if not self.isLegal(c, pos):
            return False
        elif self._fiveRow(c, pos):
            self.winner = c
            self.b.move(pos[0], pos[1], c)
            return True
        else:
            self.b.move(pos[0], pos[1], c)
            if self.movesDone == Board.BOARD_SIZE * Board.BOARD_SIZE:
                self.winner = Board.STONE_NOTHING
            return True
        
    def playToTheEnd(self, p1, p2):
        """ alternate playing moves between players until the game is over. """
        assert p1.color == -p2.color
        i = 0
        p1.game = self
        p2.game = self
        players = [p1, p2]
        while not self.gameOver():
            p = players[i]
            self.performAction(p.getAction())
            i = (i + 1) % 2
            

class FivePlayer(Agent):
    greedySelection = False
    temperature = 1.

    def __init__(self, net, game, color = Board.STONE_BLACK, **args):
        self.game = game
        self.net = net
        self.color = color
        if self.greedySelection:
            self.temperature = 0.        
        self.setArgs(**args)
        
        
    def getAction(self):
        ba = self.get_input_values(self.color, self.game.b)
        # network is given inputs with self/other as input, not black/white
        if self.color != Board.STONE_BLACK:
            tmp = np.zeros(len(ba))
            tmp[:len(ba)-1:2] = ba[1:len(ba):2]
            tmp[1:len(ba):2] = ba[:len(ba)-1:2]
            ba = tmp
        self.module.reset()
        return [self.color, self._legalizeIt(self.module.activate(ba))]
    
    
    def get_input_values(self, who, board):
        '''
        Returns:
        -----------
        vector: numpy.1darray
            the input vector
        '''
#         print('boar.stone shape: ' + str(board.stones.shape))
        v = board.stones
#         print('vectorized board shape: ' + str(v.shape))

        iv = np.zeros(v.shape[0] * 2 + 2)
  
        iv[0:v.shape[0]] = (v == Board.STONE_BLACK).astype(int)
        iv[v.shape[0]:v.shape[0] * 2] = (v == Board.STONE_WHITE).astype(int)
        iv[-2] = 1 if who == Board.STONE_BLACK else 0  # turn to black move
        iv[-1] = 1 if who == Board.STONE_WHITE else 0  # turn to white move
#         print(iv.shape)
#         print(iv)
        return iv
    
        
    def _legalizeIt(self, a):
        """ draw index from an array of values, filtering out illegal moves. """
        if not min(a) >= 0:
            print(a)
            print((min(a)))
            print((self.module.params))
            print((self.module.inputbuffer))
            print((self.module.outputbuffer))
            raise Exception('No positve value in array?')
        legals = self.game.getLegals(self.color)
        vals = np.ones(len(a))*(-100)*(1+self.temperature)
        for i in map(self.convertPosToIndex, legals):
            vals[i] = a[i]
        drawn = self.convertIndexToPos(drawGibbs(vals, self.temperature))
        assert drawn in legals
        return drawn        
    
    
    @staticmethod
    def convertIndexToPos(i):
        return divmod(i, Board.BOARD_SIZE)

    @staticmethod
    def convertPosToIndex(p):
        return p[0] * Board.BOARD_SIZE + p[1]
    
    def newEpisode(self):
        self.module.reset()
        
        
        
        
        
class RandomPlayer(Agent):
    def __init__(self, game, color = Board.STONE_BLACK, **args):
        self.game = game
        self.color = color
        self.setArgs(**args)
    
    def getAction(self):
        return [self.color, choice(self.game.getLegals(self.color))]
    

def train():    
    dim = Board.BOARD_SIZE
    g = FiveGame((dim, dim))
    
    n = ConvolutionalBoardNetwork(dim, 5, 3)
    p1 = FivePlayer(n, g)
    p2 = RandomPlayer(g)
    p2.color = g.WHITE
    episodes = 5000
    
    for i in range(episodes):
        g.playToTheEnd(p1, p2)
        g.reset()

