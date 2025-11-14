# -*- coding: utf-8 -*-
"""
Root-parallel version of MCTS for the AlphaZero Gomoku project (multiprocessing).

Idea:
-  n_workers process，each process has its own MCTS + policy_value_fn。
- Each worker deal with (n_playout / n_workers) playout。


"""

import copy
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_alphaZero import MCTS, softmax


_GLOBAL_POLICY_FN = None


def _init_worker(policy_value_fn):
    
    global _GLOBAL_POLICY_FN
    _GLOBAL_POLICY_FN = policy_value_fn


def _worker_job(args):
    """
    worker：
    - get the board 
    - Build a MCTS，use global policy_value_fn
    - 跑 n_playout 次 playout
    - 回傳 (acts, visits)
    """
    state, n_playout, c_puct, temp, seed = args


    np.random.seed(seed)
    mcts = MCTS(_GLOBAL_POLICY_FN, c_puct=c_puct, n_playout=n_playout)
    acts, _, visits = mcts.get_move_probs_and_visits(state, temp=temp)
    return acts, visits


class MCTSRootParallel(object):
    """Root-parallel MCTS using multiprocessing."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, n_workers=4):
        assert n_workers >= 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._n_workers = n_workers

    def get_move_probs(self, state, temp=1e-3):
        """
        對外接口跟原本 MCTS.get_move_probs 一樣：
        - 輸入 board state
        - 回傳 (actions, probs)
        """

        # worker=1 或 playout 數太小 → 直接用單執行緒 baseline
        if self._n_workers <= 1 or self._n_playout <= self._n_workers:
            mcts = MCTS(self._policy, c_puct=self._c_puct,
                        n_playout=self._n_playout)
            acts, act_probs, _ = mcts.get_move_probs_and_visits(state, temp)
            return acts, act_probs

        n_per_worker = self._n_playout // self._n_workers

        # 要送進去的 job list
        jobs = []
        for _ in range(self._n_workers):
 
            s = copy.deepcopy(state)
            seed = np.random.randint(0, 10**9)
            jobs.append((s, n_per_worker, self._c_puct, temp, seed))

        total_visits = defaultdict(float)

        # 建 ProcessPoolExecutor，每個 worker process 用 _init_worker 初始化 policy_value_fn
        with ProcessPoolExecutor(
            max_workers=self._n_workers,
            initializer=_init_worker,
            initargs=(self._policy,)
        ) as executor:

            for acts, visits in executor.map(_worker_job, jobs):
                for a, v in zip(acts, visits):
                    total_visits[a] += float(v)

        if not total_visits:
            return [], np.array([], dtype=np.float32)

        acts = list(total_visits.keys())
        visits_arr = np.array([total_visits[a] for a in acts], dtype=np.float32)
        act_probs = softmax(1.0 / temp * np.log(visits_arr + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        為了介面相容保留這個函式，但這個 root-parallel 版本
        每次 get_move_probs 都會重頭建一批 worker 跑樹，所以這裡就不做事。
        """
        return

    def __str__(self):
        return "MCTSRootParallelMP"


class MCTSRootParallelPlayer(object):
    """Player wrapper 使用 multiprocessing 版 root-parallel MCTS。"""

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
