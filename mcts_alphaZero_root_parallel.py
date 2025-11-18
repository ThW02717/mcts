# -*- coding: utf-8 -*-
"""
Root-parallel version of MCTS for the AlphaZero Gomoku project (multiprocessing).

Idea:
- n_workers processes, each process owns its **own persistent MCTS + policy_value_fn**.
- For each move, every worker runs (n_playout / n_workers) playouts on its local MCTS,
  then we aggregate root visit counts across workers.
- update_with_move(move) is broadcast to all workers, so each worker's tree
  can reuse information across moves (just like the original single-thread MCTS).
"""

import copy
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_alphaZero import MCTS, softmax

# ====== globals in worker processes ======
_GLOBAL_POLICY_FN = None
_GLOBAL_MCTS = None

def _init_worker(policy_value_fn, c_puct, n_playout_per_worker):
    """
    This runs ONCE in each worker process when the pool is created.
    We create a persistent MCTS here and reuse it across all calls.
    """
    global _GLOBAL_POLICY_FN, _GLOBAL_MCTS
    _GLOBAL_POLICY_FN = policy_value_fn
    _GLOBAL_MCTS = MCTS(policy_value_fn,
                        c_puct=c_puct,
                        n_playout=n_playout_per_worker)


def _worker_search(args):
    """
    Worker job for one move:
    - Take a board state copy and temp.
    - Run MCTS from the current _GLOBAL_MCTS root
      (which already includes previous search info, if any).
    - Return (acts, visits) at the root.
    """
    state, temp, seed = args

    np.random.seed(seed)

    acts, _, visits = _GLOBAL_MCTS.get_move_probs_and_visits(state, temp=temp)
    return acts, visits


def _worker_update(move):

    _GLOBAL_MCTS.update_with_move(move)
    return None


class MCTSRootParallel(object):
    """Root-parallel MCTS using multiprocessing, WITH tree reuse support."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, n_workers=4):
        assert n_workers >= 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._n_workers = n_workers


        self._single_mcts = MCTS(self._policy,
                                 c_puct=self._c_puct,
                                 n_playout=self._n_playout)


        if (self._n_workers <= 1) or (self._n_playout <= self._n_workers):
            self._executor = None
            self._n_per_worker = None
        else:
            self._n_per_worker = self._n_playout // self._n_workers
            self._executor = ProcessPoolExecutor(
                max_workers=self._n_workers,
                initializer=_init_worker,
                initargs=(self._policy, self._c_puct, self._n_per_worker),
            )

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass

    def get_move_probs(self, state, temp=1e-3):
        """
        Run root-parallel search 
        """

        if self._executor is None:
            acts, act_probs, _ = self._single_mcts.get_move_probs_and_visits(state, temp)
            return acts, act_probs


        jobs = []
        for _ in range(self._n_workers):
            s = copy.deepcopy(state)
            seed = np.random.randint(0, 10**9)
            jobs.append((s, temp, seed))

        total_visits = defaultdict(float)

        for acts, visits in self._executor.map(_worker_search, jobs):
            for a, v in zip(acts, visits):
                total_visits[a] += float(v)

        if not total_visits:
            return [], np.array([], dtype=np.float32)

        acts = list(total_visits.keys())
        visits_arr = np.array([total_visits[a] for a in acts], dtype=np.float32)
        act_probs = softmax(1.0 / temp * np.log(visits_arr + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if self._executor is None:
            self._single_mcts.update_with_move(last_move)
            return


        moves = [last_move] * self._n_workers
        list(self._executor.map(_worker_update, moves))

    def __str__(self):
        return "MCTSRootParallelMP"
    

class MCTSRootParallelPlayer(object):
    """Player wrapper"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, n_workers=4, is_selfplay=0):
        self.mcts = MCTSRootParallel(policy_value_function,
                                     c_puct=c_puct,
                                     n_playout=n_playout,
                                     n_workers=n_workers)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
   
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:

                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSRootParallelPlayerMP {}".format(self.player)
