import os

os.environ["OMP_NUM_THREADS"] = "1"       
os.environ["MKL_NUM_THREADS"] = "1"      
os.environ["NUMEXPR_NUM_THREADS"] = "1"   
os.environ["TORCH_NUM_THREADS"] = "1"
from game import Board, Game
import numpy as np
import pickle
import torch
from tqdm import tqdm
import os 
import csv
torch.set_num_threads(1) 
torch.set_num_interop_threads(1)


from mcts_pure import MCTSPlayer as OriginalAIPlayer
from mcts_pure_root_parallel import MCTSPureRootParallelPlayer

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
def run():
    seed_torch(77)
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 10
    ROUNDS = 1
    playout = 200
    baselineplayout = 200
    players = [
(   "root_2", MCTSPureRootParallelPlayer(
                             c_puct=5,
                            n_playout=playout,
                            n_workers=2,
                            time_limit=4)),
(    "root_4", MCTSPureRootParallelPlayer(
                             c_puct=5,
                            n_playout=playout,
                            n_workers=4,
                            time_limit=4)),
(    "root_8", MCTSPureRootParallelPlayer(
                             c_puct=5,
                            n_playout=playout,
                            n_workers=8,
                            time_limit=4)) 
    ]
    output_csv_path = './results'
    output_file_name = f'Efficiency_playout={playout}.csv'
    # model_file = f"./policy_8000.model"
    ##################################################################################
    os.makedirs(output_csv_path, exist_ok=True)
    output_csv_file = os.path.join(output_csv_path,output_file_name)
    csv_header = [
        'Tester_Workers',
        'Baseline',
        'Tester_winrate_Black',
        'Tester_winrate_White',
        'Baseline_winrate_Black',
        'Baseline_winrate_White',
        'total_games',
        'Tester_Win_rate',
        'baseline_avg_simulations',
        'tester_avg_simulations',

    ]    
    all_results_data = []
    progress = tqdm(players, total=len(players), desc="Workers")
    baseline = OriginalAIPlayer(c_puct=5, n_playout=baselineplayout)
    for name, player in progress:
        baseline_simulations = 0
        baseline_moves = 0
        tester_simulations = 0
        tester_moves = 0
        tester = player
        p1win = np.zeros(2)
        p2win = np.zeros(2)
        inner_progress = tqdm(range(1, ROUNDS + 1), desc="Games", leave=False)
        for round in inner_progress:
            for who_start in range(2):
                board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=5)
                game = Game(board)
                tester.reset_player()
                winner, _, sim_cnt, move_cnt = game.start_play(player1=tester, player2=baseline, start_player=who_start, is_shown=False)
                baseline_simulations += sim_cnt[baseline.player]
                baseline_moves += move_cnt[baseline.player]
                tester_simulations += sim_cnt[tester.player]
                tester_moves += move_cnt[tester.player]
                if winner == tester.player:
                    if who_start == 0: # player1 first
                        p1win[0] += 1 # win[0]表示先手獲勝次數
                    elif who_start ==1:
                        p1win[1] += 1
                elif winner == baseline.player:
                    if who_start == 0:
                        p2win[1] += 1
                    elif who_start ==1:
                        p2win[0] += 1

        all_results_data.append({
        'Tester_Workers': name,
        'Baseline': baseline,
        'Tester_winrate_Black': p1win[0],
        'Tester_winrate_White': p1win[1],
        'Baseline_winrate_Black': p2win[0],
        'Baseline_winrate_White': p2win[1],
        'total_games': ROUNDS*2,
        'Tester_Win_rate': f"{(p1win[0]+p1win[1])/(ROUNDS*2):.2%}",
        'baseline_avg_simulations': f"{(baseline_simulations / baseline_moves):.2f}" if baseline_moves > 0 else "0",
        'tester_avg_simulations': f"{(tester_simulations / tester_moves):.2f}" if tester_moves > 0 else "0"
    })
        print("\n===== Final Battle Report =====")
        print("Winrate:")
        print(f"Tester_{name}:{(p1win[0]+p1win[1])}/{(ROUNDS*2)} (In Black: {p1win[0]}/{ROUNDS}, In White: {p1win[1]}/{ROUNDS}, avg simulations: {tester_simulations / tester_moves if tester_moves > 0 else 0:.2f})")
        print(f"Baseline:{(p2win[0]+p2win[1])}/{(ROUNDS*2)} (In Black: {p2win[0]}/{ROUNDS}, In White: {p2win[1]}/{ROUNDS}, avg simulations: {baseline_simulations / baseline_moves if baseline_moves > 0 else 0:.2f})")
        print("===============================")
    print(f"\nAll battles finished. Writing results to {output_csv_path}...")
    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(all_results_data)
        print("Successfully wrote results to CSV.")
    except IOError:
        print(f"Error writing to {output_csv_file}.")
if __name__ == '__main__':
    run()