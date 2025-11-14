# -*- coding: utf-8 -*-
"""
Root-parallel MCTS using multiple *processes* (not threads).

- 核心 MCTS 還是用 mcts_alphaZero.MCTS（單執行緒）
- 這裡只負責：
    - 開 n_workers 個 process
    - 每個 process 建一個獨立 MCTS 從同一個 root state 出發
    - 跑 n_playout / n_workers 次 playout
    - 把 root children 的 visit counts 合併
"""

import copy
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_alphaZero import MCTS, softmax

# === 全域變數：給 worker process 用 ===
# 在 Linux 上，用 fork 的話，這些東西會在 fork 時一起被複製過去，
# 不需要透過 pickle 傳遞，避免把整個 model/FN 一直序列化。
_GLOBAL_POLICY_FN = None
_GLOBAL_C_PUCT = None
_GLOBAL_N_PLAYOUT_PER_WORKER = None


def _set_globals(policy_value_fn, c_puct, n_playout_per_worker):
    """
    在主 process 設定全域參數。
    在 Linux + fork 的情況下，之後新開的 process 會複製這份狀態。
    """
    global _GLOBAL_POLICY_FN, _GLOBAL_C_PUCT, _GLOBAL_N_PLAYOUT_PER_WORKER
    _GLOBAL_POLICY_FN = policy_value_fn
    _GLOBAL_C_PUCT = c_puct
    _GLOBAL_N_PLAYOUT_PER_WORKER = n_playout_per_worker


def _worker_job(args):
    """
    單個 worker process 要做的事：
    - 從 leaf state 出發建立一個 MCTS 實例
    - 跑固定次數的 playout
    - 回傳：該 worker 看到的 (acts, visits)
    """
    state, temp, seed = args

    # 每個 process 自己 seed，避免完全走一模一樣的路
    np.random.seed(seed)

    # 用全域的 policy_value_fn, c_puct, n_playout_per_worker
    mcts = MCTS(
        policy_value_fn=_GLOBAL_POLICY_FN,
        c_puct=_GLOBAL_C_PUCT,
        n_playout=_GLOBAL_N_PLAYOUT_PER_WORKER,
    )

    # 用你在 mcts_alphaZero 加的這個 API
    acts, _, visits = mcts.get_move_probs_and_visits(state, temp=temp)

    # 為了防止跨 process 傳遞 issue，統一轉成基本型別
    return list(acts), [float(v) for v in visits]


class MCTSRootParallelMP(object):
    """Root-parallel wrapper around original MCTS using multiple processes."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, n_workers=4):
        assert n_workers >= 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._n_workers = n_workers

    def get_move_probs(self, state, temp=1e-3):
        """
        對外介面跟 MCTS.get_move_probs 一樣：
        - 輸入 board state
        - 回傳 (actions, probs)
        """
        # 太小就不要平行了，直接用原本 MCTS
        if self._n_workers <= 1 or self._n_playout <= self._n_workers:
            mcts = MCTS(self._policy, c_puct=self._c_puct,
                        n_playout=self._n_playout)
            acts, act_probs, _ = mcts.get_move_probs_and_visits(state, temp=temp)
            return acts, act_probs

        n_per_worker = self._n_playout // self._n_workers

        # 設好 global，之後 fork 的 process 會 copy 到
        _set_globals(self._policy, self._c_puct, n_per_worker)

        tasks = []
        total_visits = defaultdict(float)

        # 這裡用 ProcessPoolExecutor → 多個 process，不會被 GIL 卡死
        with ProcessPoolExecutor(max_workers=self._n_workers) as ex:
            for _ in range(self._n_workers):
                state_copy = copy.deepcopy(state)
                seed = np.random.randint(0, 10**9)
                tasks.append(ex.submit(_worker_job, (state_copy, temp, seed)))

            for fut in tasks:
                acts, visits = fut.result()
                for a, v in zip(acts, visits):
                    total_visits[a] += v

        if not total_visits:
            return [], np.array([], dtype=np.float32)

        acts = list(total_visits.keys())
        visits_arr = np.array([total_visits[a] for a in acts], dtype=np.float32)
        act_probs = softmax(1.0 / temp * np.log(visits_arr + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        這個 root-parallel 版本每一步都是重新開一批 worker，
        沒有 reuse 之前的樹，所以這裡其實可以什麼都不做。
        """
        return

    def __str__(self):
        return "MCTSRootParallelMP"


class MCTSRootParallelMPPlayer(object):
    """Player wrapper 使用 root-parallel (multi-process) MCTS。"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, n_workers=4, is_selfplay=0):
        self.mcts = MCTSRootParallelMP(policy_value_function,
                                       c_puct=c_puct,
                                       n_playout=n_playout,
                                       n_workers=n_workers)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # 這版 root-parallel 每步都新建 search，不保留樹
        return

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
            else:
                move = np.random.choice(acts, p=probs)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSRootParallelMP {}".format(self.player)
