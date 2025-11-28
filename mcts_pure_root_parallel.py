# -*- coding: utf-8 -*-
"""
Root-parallel MCTS for the "pure" (rollout-based) MCTS
Supports:
- fixed n_playout
- time_limit (sec)
"""

import time
import copy
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_pure import MCTS, policy_value_fn


def _worker_job(args):
    """
    Worker job: run `n_playout` playouts on a fresh MCTS, return visit counts.
    """
    state, n_playout, c_puct, seed = args

    np.random.seed(seed)

    # create local MCTS
    mcts = MCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout)

    # do search
    state_copy = copy.deepcopy(state)
    _ = mcts.get_move(state_copy)

    if not mcts._root._children:
        return [], []

    act_visits = [(act, node._n_visits)
                  for act, node in mcts._root._children.items()]
    acts, visits = zip(*act_visits)
    return list(acts), list(visits)


class MCTSPureRootParallel(object):

    def __init__(self, c_puct=5, n_playout=2000, n_workers=4, time_limit=None):
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._n_workers = n_workers
        self._time_limit = time_limit

        # fallback single-thread version
        self._single_mcts = MCTS(policy_value_fn, c_puct, n_playout)

        if (n_workers <= 1) or (n_playout <= n_workers):
            self._executor = None
            self._n_per_worker = None
        else:
            self._n_per_worker = n_playout // n_workers
            self._executor = ProcessPoolExecutor(max_workers=n_workers)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except:
            pass

    # ============================================================
    # get_move() with support for time-limit
    # ============================================================
    def get_move(self, state):
        """Run root-parallel MCTS and pick the action with highest visit count."""

        # --- case 1: no parallel ---
        if self._executor is None:
            if self._time_limit is None:
                return self._single_mcts.get_move(state)  # (move)
            else:
                # time-limit sequential fallback
                return self._run_single_with_time(state)

        # --- case 2: parallel root search ---
        total_visits = defaultdict(float)
        simulation_count = 0

        start_time = time.time()
        use_time = self._time_limit is not None
        deadline = start_time + (self._time_limit or 0)

        while True:
            # stop by time or playout count
            if use_time:
                if time.time() >= deadline:
                    break
            else:
                if simulation_count >= self._n_playout:
                    break

            # schedule workers
            jobs = []
            for _ in range(self._n_workers):
                s = copy.deepcopy(state)
                seed = np.random.randint(0, 10**9)
                jobs.append((s, self._n_per_worker, self._c_puct, seed))

            # gather from workers
            for acts, visits in self._executor.map(_worker_job, jobs):
                for a, v in zip(acts, visits):
                    total_visits[a] += float(v)

            simulation_count += self._n_workers * self._n_per_worker

        if not total_visits:
            raise RuntimeError("Pure Root Parallel: no visits returned.")

        # pick the action with max visits
        best_act, _ = max(total_visits.items(), key=lambda x: x[1])
        return best_act

    def _run_single_with_time(self, state):
        """Sequential fallback for time-limit mode."""
        mcts = self._single_mcts
        simulation_count = 0
        deadline = time.time() + self._time_limit

        while time.time() < deadline:
            s = copy.deepcopy(state)
            mcts._playout(s)
            simulation_count += 1

        return mcts.get_move(state)

    def update_with_move(self, last_move):
        # Pure MCTS: no tree reuse → no-op
        return

    def __str__(self):
        return "MCTSPureRootParallel"


class MCTSPureRootParallelPlayer(object):

    def __init__(self, c_puct=5, n_playout=2000, n_workers=4, time_limit=None):
        self.mcts = MCTSPureRootParallel(
            c_puct=c_puct,
            n_playout=n_playout,
            n_workers=n_workers,
            time_limit=time_limit,
        )

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # pure version: no-op
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) == 0:
            print("WARNING: board is full")
            return None
        move = self.mcts.get_move(board)
        self.mcts.update_with_move(-1)
        return move

    def __str__(self):
        return "MCTSPureRootParallelPlayer {}".format(self.player)
