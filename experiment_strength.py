# experiment_strength.py
import numpy as np

from game import Board, Game
from policy_value_net import PolicyValueNet
from mcts_alphaZero import MCTSPlayer
from mcts_alphaZero_root_parallel import MCTSRootParallelPlayer


if __name__ == "__main__":
    # TODO: 改成跟 human_play.py 一樣
    width, height, n_in_row = 10, 10, 5

    board = Board(width=width, height=height, n_in_row=n_in_row)
    board.init_board() 
    game = Game(board)

    policy_net = PolicyValueNet(width, height, model_file='policy_8000.model')

    N_PLO = 6400

    baseline = MCTSPlayer(policy_net.policy_value_fn,
                          c_puct=5,
                          n_playout=N_PLO,
                          is_selfplay=0)

    rootp = MCTSRootParallelPlayer(policy_net.policy_value_fn,
                                   c_puct=5,
                                   n_playout=N_PLO,
                                   n_workers=4,
                                   is_selfplay=0)

    n_games = 30
    win_root, win_base, tie = 0, 0, 0

    for i in range(n_games):
        # 偶數局 root-parallel 先手，奇數局 baseline 先手，公平一點
        if i % 2 == 0:
            winner = game.start_play(rootp, baseline, start_player=0, is_shown=0)
            # 這裡假設 winner 回傳 1/2/-1 (看你的 game 實作，如果不一樣就自己修一下)
            if winner == 1:
                win_root += 1
            elif winner == 2:
                win_base += 1
            else:
                tie += 1
        else:
            winner = game.start_play(baseline, rootp, start_player=0, is_shown=0)
            if winner == 1:
                win_base += 1
            elif winner == 2:
                win_root += 1
            else:
                tie += 1

        baseline.reset_player()
        rootp.reset_player()

    print("Total games:", n_games)
    print("Root-parallel wins :", win_root)
    print("Baseline wins      :", win_base)
    print("Ties               :", tie)
