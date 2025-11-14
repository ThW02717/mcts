import os

os.environ["OMP_NUM_THREADS"] = "1"       
os.environ["MKL_NUM_THREADS"] = "1"      
os.environ["NUMEXPR_NUM_THREADS"] = "1"   
os.environ["TORCH_NUM_THREADS"] = "1"
# experiment_speed.py
import time
import numpy as np

from game import Board
from policy_value_net import PolicyValueNet
from mcts_alphaZero import MCTSPlayer
from mcts_alphaZero_root_parallel import MCTSRootParallelPlayer


def measure_once(player, board):
    t0 = time.perf_counter()
    player.mcts.get_move_probs(board, temp=1e-3)
    t1 = time.perf_counter()
    return t1 - t0


def measure_avg(player, board, trials=5):
    times = []
    for _ in range(trials):
        player.reset_player()
        times.append(measure_once(player, board))
    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":

    width, height, n_in_row = 10, 10, 5
    board = Board(width=width, height=height, n_in_row=n_in_row)
    board.init_board()  

    policy_net = PolicyValueNet(width, height, model_file='policy_8000.model')
    # play out: 1600 6400 12800
    N_PLO = 1600

    baseline = MCTSPlayer(policy_net.policy_value_fn,
                          c_puct=5,
                          n_playout=N_PLO,
                          is_selfplay=0)

    root2 = MCTSRootParallelPlayer(policy_net.policy_value_fn,
                                   c_puct=5,
                                   n_playout=N_PLO,
                                   n_workers=2,
                                   is_selfplay=0)

    root4 = MCTSRootParallelPlayer(policy_net.policy_value_fn,
                                   c_puct=5,
                                   n_playout=N_PLO,
                                   n_workers=4,
                                   is_selfplay=0)
    root8 = MCTSRootParallelPlayer(policy_net.policy_value_fn,
                                   c_puct=5,
                                   n_playout=N_PLO,
                                   n_workers=8,
                                   is_selfplay=0)

    for name, p in [("baseline", baseline),
                    ("root-2", root2),
                    ("root-4", root4),
                    ("root-8", root8)]:
        mean_t, std_t = measure_avg(p, board, trials=5)
        print(f"{name:8s}  avg = {mean_t:.4f} s   std = {std_t:.4f} s")
