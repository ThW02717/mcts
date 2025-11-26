import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import time
import numpy as np

from game import Board
from mcts_pure import MCTSPlayer
from mcts_pure_root_parallel import MCTSPureRootParallelPlayer


def measure_once(player, board):
    """
    Measure one call of MCTS get_move(board).
    Pure MCTS does NOT return move_probs, only `move`.
    So we time only `player.mcts.get_move(board)`.
    """
    t0 = time.perf_counter()
    player.mcts.get_move(board)
    t1 = time.perf_counter()
    return t1 - t0


def measure_avg(player, board, trials=40):
    """
    Run multiple trials and report avg / std.
    Pure MCTS does not reuse tree, so no need to update_with_move.
    But for consistency, we still call reset_player().
    """
    times = []
    for _ in range(trials):
        player.reset_player()
        times.append(measure_once(player, board))
    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":

    width, height, n_in_row = 10, 10, 5
    board = Board(width=width, height=height, n_in_row=n_in_row)
    board.init_board()

    # play out: 1600 6400 12800 可自行設定
    N_PLO = 1600

    # baseline (single-thread)
    baseline = MCTSPlayer(c_puct=5, n_playout=N_PLO)

    # root-parallel versions
    root2 = MCTSPureRootParallelPlayer(c_puct=5,
                                       n_playout=N_PLO,
                                       n_workers=2)

    root4 = MCTSPureRootParallelPlayer(c_puct=5,
                                       n_playout=N_PLO,
                                       n_workers=4)

    root8 = MCTSPureRootParallelPlayer(c_puct=5,
                                       n_playout=N_PLO,
                                       n_workers=8)

    for name, p in [
        ("baseline", baseline),
        ("root-2",  root2),
        ("root-4",  root4),
        ("root-8",  root8),
    ]:
        mean_t, std_t = measure_avg(p, board, trials=40)
        print(f"{name:8s}  avg = {mean_t:.4f} s   std = {std_t:.4f} s")
