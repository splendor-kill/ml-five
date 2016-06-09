from math import log, sqrt
from random import choice
import time

from tentacle.board import Board
from tentacle.game import Game


class MonteCarlo(object):
    def __init__(self, **kwargs):

        self.states = []
        self.wins = {}
        self.plays = {}

        self.max_depth = 0
        self.stats = {}

        self.calculation_time = float(kwargs.get('time', 1))
        self.max_moves = int(kwargs.get('max_moves', Board.BOARD_SIZE ** 2))

        # Exploration constant, increase for more exploratory moves,
        # decrease to prefer moves with known higher win rates.
        self.C = float(kwargs.get('C', 1.4))
        
    
    def select(self, board, moves, who, **kwargs):
#         the_game = kwargs.get('context')

        if Game.on_training:
            self.calculation_time = 60
        
        self.max_depth = 0
        self.stats = {}

#         state = self.states[-1]
        player = who  # board.current_player(state)
        legal = moves  # board.legal_plays(self.states[:])

        # Bail out early if there is no real choice to be made.
        if not legal:
            return
        if len(legal) == 1:
            return legal[0]

        games = 0
        begin = time.time()
        
        while time.time() - begin < self.calculation_time:
            self.sim(board, player)
            games += 1
#             if games > 0:
#                 break
        
#         print('states size: ', len(self.states))
        
        self.stats.update(games=games, max_depth=self.max_depth, time=str(time.time() - begin))
        print(len(self.plays), self.stats['games'], self.stats['time'])
#         print("Maximum depth searched:", self.max_depth)

        moves_states = moves  # [(p, board.next_state(state, p)) for p in legal]

        # Display the stats for each possible play.
        self.stats['moves'] = sorted(
            ({'move': divmod(Board.change(board, S), Board.BOARD_SIZE),
              'percent': 100 * self.wins.get((player, str(S)), 0) / self.plays.get((player, str(S)), 1),
              'wins': self.wins.get((player, str(S)), 0),
              'plays': self.plays.get((player, str(S)), 0)}
             for S in moves_states),
            key=lambda x: (x['percent'], x['plays']),
            reverse=True
        )
        
#         print('moves')
#         for m in moves_states:
#             print(m)
#         print()
#         print('wins')
#         for m, c in self.wins:
#             print(m, c)
        
            
        
#         for m in self.stats['moves']:
#             print("{move}: {percent:.2f}% ({wins} / {plays})".format(**m))


        
        # Pick the move with the highest percentage of wins.
#         percent_wins, num_moves, move = max(
#             (self.wins.get((player, str(S)), 0) / 
#              self.plays.get((player, str(S)), 1),
#              self.plays.get((player, str(S)), 0),
#              S)
#             for S in moves_states
#         )
        
        move = max(moves_states, key=lambda S: (self.wins.get((player, str(S)), 0) / self.plays.get((player, str(S)), 1), self.plays.get((player, str(S)), 0)))

        return move
    
    

    def sim(self, board, who):
        plays, wins = self.plays, self.wins

        visited_path = []
#         states_copy = self.states[:]
        state = board  # states_copy[-1]
        player = who  # board.current_player(state)

        expand = True
        winner = Board.STONE_NOTHING
        for t in range(1, self.max_moves + 1):
#             plt.matshow(state.stones.reshape(-1, Board.BOARD_SIZE))
            moves, player = Game.possible_moves(state)
            
            # legal = board.legal_plays(states_copy)
#             moves_states = moves  # [(p, board.next_state(state, p)) for p in legal]

            if all(plays.get((player, str(S))) for S in moves):
                log_total = log(sum(plays[(player, str(S))] for S in moves))
                state_new = max(moves, key=lambda S: wins[(player, str(S))] / plays[(player, str(S))] + self.C * sqrt(log_total / plays[(player, str(S))]))
#                 value, state_new = max((wins[(player, str(S))] / plays[(player, str(S))] + self.C * sqrt(log_total / plays[(player, str(S))]), S) for S in moves)
            else:
                state_new = choice(moves)

            over, winner, _ = state_new.is_over(state)
            
            state = state_new

#             states_copy.append(state)

            # `player` here and below refers to the player
            # who moved into that particular state.
            if expand and (player, str(state)) not in plays:
#                 expand = False
                plays[(player, str(state))] = 0
                wins[(player, str(state))] = 0
                if t > self.max_depth:
                    self.max_depth = t

            visited_path.append((player, str(state)))


#             player = Board.oppo(player)#board.current_player(state_new)
#             winner = board.winner(states_copy)
            if over:
                break

        for player, state in visited_path:
            if (player, state) not in plays:
                continue
            plays[(player, state)] += 1
            if player == winner:
                wins[(player, state)] += 1
        
#         print(len(plays), len(wins))        

    def update(self, state):
        self.states.append(state)
        
    
    
