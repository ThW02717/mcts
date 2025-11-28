# -*- coding: utf-8 -*-
"""
Root-parallel MCTS (pure, rollout-based) with:
- n_playout or time_limit stopping condition
- returns (best_move, simulation_count)

"""

import copy
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_pure import MCTS, policy_value_fn   



def _worker_job(args):

    state, n_playout, c_puct, seed = args

    np.random.seed(seed)

    mcts = MCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout)

    state_copy = copy.deepcopy(state)
    _ = mcts.get_move(state_copy)

    if not mcts._root._children:
        return [], []

    act_visits = [(act, node._n_visits)
                  for act, node in mcts._root._children.items()]
    acts, visits = zip(*act_visits)
    return list(acts), list(visits)


class MCTSPureRootParallel(object):


    def __init__(self, c_puct=5, n_playout=10000,
                 n_workers=4, time_limit=None):

        self._c_puct = c_puct
        self._n_playout = int(n_playout)
        self._n_workers = int(max(1, n_workers))
        self._time_limit = time_limit


        self._single_mcts = MCTS(policy_value_fn, c_puct, n_playout)

        if (self._n_workers <= 1):

            self._executor = None
        else:

            self._chunk_n_per_worker = max(1, self._n_playout // self._n_workers)
            self._executor = ProcessPoolExecutor(max_workers=self._n_workers)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _run_one_batch(self, state, n_playout_per_worker):
     
        jobs = []
        for _ in range(self._n_workers):
            s = copy.deepcopy(state)
            seed = np.random.randint(0, 10**9)
            jobs.append((s, n_playout_per_worker, self._c_puct, seed))

        total_visits = defaultdict(float)

        for acts, visits in self._executor.map(_worker_job, jobs):
            for a, v in zip(acts, visits):
                total_visits[a] += float(v)

        if not total_visits:
            return [], [], 0

        acts = list(total_visits.keys())
        visits = [total_visits[a] for a in acts]
        sims_this_batch = n_playout_per_worker * self._n_workers
        return acts, visits, sims_this_batch

    def get_move(self, state):

        if self._executor is None:
            move, simulation_count = self._single_mcts.get_move(copy.deepcopy(state))
            return move, simulation_count

        total_visits = defaultdict(float)
        simulation_count = 0


        if self._time_limit is not None:
            start_time = time.time()
   
            while True:
                now = time.time()
                if (now - start_time) >= self._time_limit and simulation_count > 0:
                    break

                acts, visits, sims = self._run_one_batch(
                    state, self._chunk_n_per_worker
                )
                if sims == 0:
                    break

                simulation_count += sims
                for a, v in zip(acts, visits):
                    total_visits[a] += v

                if (time.time() - start_time) >= self._time_limit:
                    break

        else:
            while simulation_count < self._n_playout:
                sims_left = self._n_playout - simulation_count

                n_per_worker = min(
                    self._chunk_n_per_worker,
                    max(1, sims_left // self._n_workers)
                )

                acts, visits, sims = self._run_one_batch(
                    state, n_per_worker
                )
                if sims == 0:
                    break

                simulation_count += sims
                for a, v in zip(acts, visits):
                    total_visits[a] += v

                if sims_left <= self._n_workers:
                    break

        if not total_visits:
            avail = getattr(state, "availables", None)
            if avail:
                rand_move = np.random.choice(avail)
                return rand_move, simulation_count

            raise RuntimeError("MCTSPureRootParallel: no visits and no availables")
        best_move, _ = max(total_visits.items(), key=lambda x: x[1])
        return best_move, simulation_count

    def update_with_move(self, last_move):
      
        return

    def __str__(self):
        return "MCTSPureRootParallelTimed"
    

class MCTSPureRootParallelPlayer(object):
    

    def __init__(self, c_puct=5, n_playout=10000,
                 n_workers=4, time_limit=None):
        self.mcts = MCTSPureRootParallel(
            c_puct=c_puct,
            n_playout=n_playout,
            n_workers=n_workers,
            time_limit=time_limit,
        )

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move, sim_count = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move, sim_count
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSPureRootParallelPlayer {}".format(self.player)
