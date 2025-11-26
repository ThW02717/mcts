# -*- coding: utf-8 -*-
"""
Root-parallel MCTS for the "pure" (rollout-based) MCTS version.

Idea:
- 原本 MCTS.get_move(state) 是在單一 process 裡跑 n_playout 次 playout，
  然後根據 root node child 的 visit 次數選擇動作。
- 這裡改成 root-parallel：
    * 開 n_workers 個 process
    * 每個 worker 各自 new 一個 MCTS (pure)，在 local 樹上跑 (n_playout / n_workers) 次 playout
    * 每個 worker 回傳自己 root 的 (acts, visits)
    * 主 process 把 visits 加總後，用總 visit 最多的 action 當結果
- 注意：pure MCTS 版本本來就「每手 reset 樹」，沒有跨手 tree reuse，
  因此這裡也不用實作 update_with_move 的複雜版本。
"""

import copy
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from mcts_pure import MCTS, policy_value_fn


def _worker_job(args):
    """
    單一 worker 做一手棋的工作：

    輸入：
      - state: 這一步要搜尋的盤面（呼叫端已經 deepcopy 好）
      - n_playout: 在這個 worker 內要跑幾次 playout
      - c_puct: UCT 探索係數
      - seed: 亂數種子，避免每個 worker 完全同步

    流程：
      - 在 worker 裡 new 一個 local MCTS (pure 版)
      - 用 get_move(state_copy) 跑 n_playout 次 playout（會更新 local root）
      - 回傳 local root 下每個 action 的 visit 數
        （之後由主 process 把所有 worker 的 visits 加總）
    """
    state, n_playout, c_puct, seed = args

    np.random.seed(seed)

    # 每個 worker 自己的 MCTS（只在此 worker 存在）
    mcts = MCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout)

    # get_move 會在這個 state 上跑 n_playout 次 playout
    state_copy = copy.deepcopy(state)
    _ = mcts.get_move(state_copy)

    # 如果 root 沒 children（理論上很少見），就回傳空
    if not mcts._root._children:
        return [], []

    act_visits = [(act, node._n_visits)
                  for act, node in mcts._root._children.items()]
    acts, visits = zip(*act_visits)
    return list(acts), list(visits)


class MCTSPureRootParallel(object):
    """
    Root-parallel 版的 pure MCTS。

    - 單線程 (n_workers <= 1 或 n_playout 太小) 時，直接用原本單一 MCTS。
    - 多 worker 時：
        * n_playout 平均切給每個 worker
        * 收集 worker 回傳的 visits
        * 用 aggregated visits 最大的 action 當結果
    """

    def __init__(self, c_puct=5, n_playout=2000, n_workers=4):
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._n_workers = n_workers

        # 單線程 fallback：直接沿用原本的 MCTS 實作
        self._single_mcts = MCTS(policy_value_fn, c_puct, n_playout)

        # 條件不適合平行就直接退化成單線程
        if (self._n_workers <= 1) or (self._n_playout <= self._n_workers):
            self._executor = None
            self._n_per_worker = None
        else:
            self._n_per_worker = self._n_playout // self._n_workers
            self._executor = ProcessPoolExecutor(max_workers=self._n_workers)

    def close(self):
        """手動關閉 process pool，用完這個物件時可以呼叫。"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_move(self, state):
        """
        跟原本 MCTS.get_move(state) 的介面一樣：

        - 單線程：
            直接用 self._single_mcts.get_move(state)
        - 多 worker：
            讓每個 worker 跑 n_per_worker playout，
            並把所有 worker 的 visits 加總後，選 visit 數最多的 action。
        """
        # 單線程 fallback
        if self._executor is None:
            return self._single_mcts.get_move(state)

        # 平行：組 job list
        jobs = []
        for _ in range(self._n_workers):
            s = copy.deepcopy(state)
            seed = np.random.randint(0, 10**9)
            jobs.append((s, self._n_per_worker, self._c_puct, seed))

        total_visits = defaultdict(float)

        # 收集每個 worker 的 (acts, visits) 並加總
        for acts, visits in self._executor.map(_worker_job, jobs):
            for a, v in zip(acts, visits):
                total_visits[a] += float(v)

        # 防呆：空的話就隨便從合法手挑一個
        if not total_visits:
            # 這裡不能直接拿 board.availables（手邊沒有 board）
            # 實務上幾乎不會進來，如果真的進來，可以 raise 或 random 掉
            # 這邊先選擇 raise，方便 debug。
            raise RuntimeError("MCTSPureRootParallel: no visits collected from workers.")

        # 選 aggregated visit 數最多的 action
        best_act, _ = max(total_visits.items(), key=lambda x: x[1])
        return best_act

    def update_with_move(self, last_move):
        """
        為了和原本 MCTS 介面一致而保留，但 pure 版本本來就每手 reset，
        所以這裡不做任何事（保持 stateless）。
        """
        return

    def __str__(self):
        return "MCTSPureRootParallel"


class MCTSPureRootParallelPlayer(object):
    """
    Player wrapper，介面盡量模仿原本 MCTSPlayer：

    - get_action(board)：
        * 呼叫 root-parallel 版的 get_move(board)
        * 沒有 self-play / tree reuse，跟原本 pure 版行為一致
    """

    def __init__(self, c_puct=5, n_playout=2000, n_workers=4):
        self.mcts = MCTSPureRootParallel(c_puct=c_puct,
                                         n_playout=n_playout,
                                         n_workers=n_workers)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # 保留介面，實際上 pure 版沒在用 tree reuse
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """
        和原本 MCTSPlayer.get_action(board) 相同：

        - 若有合法手，呼叫 mcts.get_move(board) 拿到一手
        - pure 版預設每手都是新樹，所以不做跨手更新
        """
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            # pure 版 design：每手都是全新搜尋 → update_with_move(-1) 是 no-op
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSPureRootParallelPlayer {}".format(self.player)
