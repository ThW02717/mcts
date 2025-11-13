# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet  # Pytorch
import numpy as np
import pickle


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def expected_score(Ri, Rj):
    return 1.0 / (1.0 + 10 ** ((Rj - Ri) / 400.0))

def run():
    n = 5
    width, height = 10, 10
    path = 'tmp/'
    policies_names = [f'policy_{i}.model' for i in range(500, 5001, 500)]
    print(f'policies:\n{policies_names}\n--------------------------------------')

    num_rounds = 50
    K = 20
    players = []

    for policy_name in policies_names:
        policy = PolicyValueNet(width, height, model_file = path + policy_name, use_gpu=True)
        player = MCTSPlayer(policy.policy_value_fn, c_puct=5, n_playout=400)
        players.append(player)

    num_players = len(players)
    elos = [1200 for i in range(num_players)]
    games_states_history = []
    games_players_history = []
    

    for round in range(1, num_rounds + 1):
        # record the scores for a round
        actual_scores = [0. for i in range(num_players)]
        expected_scores = [0. for i in range(num_players)]

        for who_start in range(2):
            for player_i in range(num_players):
                for player_j in range(player_i + 1, num_players):
                    board = Board(width=width, height=height, n_in_row=n)
                    game = Game(board) 

                    winner, states = game.start_play(players[player_i], players[player_j], start_player=who_start, is_shown=False)
                    games_states_history = games_states_history + states
                    games_players_history = games_players_history + [(player_i, player_j) for i in range(len(states))]

                    if winner == 1:
                        # player_i won
                        actual_scores[player_i] += 1.
                    elif winner == 2:
                        # player j won
                        actual_scores[player_j] += 1.
                    elif winner == -1:
                        actual_scores[player_i] += .5
                        actual_scores[player_j] += .5
                    else:
                        print(winner)
                        assert 1==0

                    # expected score
                    expected_scores[player_i] += expected_score(elos[player_i], elos[player_j])
                    expected_scores[player_j] += expected_score(elos[player_j], elos[player_i])
        
        for player in range(num_players):
            elos[player] += K * (actual_scores[player] - expected_scores[player])
                    

        print(f'round {round} finished.')
        print(f'elos:\n{elos}')

        if round % 10 == 0:
            np.save(f'tmp/games_states_history_round{round}.npy', games_states_history, allow_pickle=True)
            with open(f"tmp/games_players_history_round{round}.pkl", "wb") as f:
                pickle.dump(games_players_history, f)

    np.save('tmp/games_states_history.npy', games_states_history, allow_pickle=True)
    with open("tmp/games_players_history.pkl", "wb") as f:
        pickle.dump(games_players_history, f)

if __name__ == '__main__':
    run()
