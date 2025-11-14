from __future__ import print_function
import pickle
from game import Board, Game
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from player_agent import OriginalAIPlayer, DistilledAIPlayer_dou, DistilledAIPlayer_ResNet, DistilledAIPlayer_DouVal,DistilledAIPlayer_ResNet_Val
from tqdm import tqdm
import os 
import csv
def expected_score(Ri, Rj):
    return 1.0 / (1.0 + 10 ** ((Rj - Ri) / 400.0))
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
def run():
    
    seed_torch(77)
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 10
    SKILL_LEVELS = ['500','1000','2500','3000','3500','5000','7000','8000']
    FILE_NAME = ['500','1000','2500','3000','3500','5000','7000','8000']
    MAIA_TEST_LEVEL = ['500','1000','2500','3000','3500','5000','7000','8000']
    ROUNDS = 15
    maia_playout = 200
    output_csv_path = './results'
    output_file_name = f'alldataset_playout={maia_playout}.csv'
    ##################################################################################
    os.makedirs(output_csv_path, exist_ok=True)
    output_csv_file = os.path.join(output_csv_path,output_file_name)
    csv_header = [
        'Maia_skill_level',
        'teacher_skill_level',
        'Maia_winrate_Black',
        'Maia_winrate_White',
        'Teacher_winrate_Black',
        'Teacher_winrate_White',
        'total_games',
        'Maia_win_rate'
    ]    
    all_results_data = []
    games_states_history = []
    games_players_history = []
    progress = tqdm(MAIA_TEST_LEVEL, total=len(MAIA_TEST_LEVEL), desc="Maia Levels")
    for maia_skill_level in progress:
        outer_progress = tqdm(zip(SKILL_LEVELS, FILE_NAME), total=len(SKILL_LEVELS), desc="Teacher Levels", leave=False)
        for skill_level, file_part in outer_progress:
            
            model_file = f"./tmp/policy_{file_part}.model"
                        
            teacher = OriginalAIPlayer(model_file, BOARD_WIDTH, BOARD_HEIGHT, n_playout=400, use_gpu=True,print_outputs=False, temp=0.2)

            maia = DistilledAIPlayer_DouVal('./model/best.pth', BOARD_WIDTH, BOARD_HEIGHT, SKILL_LEVELS, 
                                    fixed_skill_level=maia_skill_level, fixed_n_playout=maia_playout, use_gpu=True, print_outputs=False,temp=0.2)

            #maia = OriginalAIPlayer(f'./tmp/policy_{maia_skill_level}.model', BOARD_WIDTH, BOARD_HEIGHT, n_playout=400, use_gpu=True,print_outputs=False, temp=0.2)
            
            p1win = np.zeros(2)
            p2win = np.zeros(2)
            inner_progress = tqdm(range(1, ROUNDS + 1), desc="Games", leave=False)
            for round in inner_progress:
                for who_start in range(2):
                    board = Board(width=BOARD_WIDTH, height=BOARD_WIDTH, n_in_row=5)
                    game = Game(board) 
                    winner, states = game.start_play(player1=teacher, player2=maia, start_player=who_start, is_shown=False)
                    if winner == teacher.player:
                        #print(f"Round {round} - {skill_level} - Teacher wins")
                        if who_start == 0: # player1 first
                            p1win[0] += 1 # win[0]表示先手獲勝次數
                        elif who_start ==1:
                            p1win[1] += 1
                    elif winner == maia.player:
                        #print(f"Round {round} - {skill_level} - Maia wins")
                        if who_start == 0:
                            p2win[1] += 1
                        elif who_start ==1:
                            p2win[0] += 1

                    games_states_history = games_states_history + states
                    games_players_history = games_players_history + [(skill_level, maia_skill_level) for i in range(len(states))]
            all_results_data.append({
            'Maia_skill_level': maia_skill_level,
            'teacher_skill_level': skill_level,
            'Maia_winrate_Black': p2win[0],
            'Maia_winrate_White': p2win[1],
            'Teacher_winrate_Black': p1win[0],
            'Teacher_winrate_White': p1win[1],
            'total_games': ROUNDS*2,
            'Maia_win_rate': f"{(p2win[0]+p2win[1])/(ROUNDS*2):.2%}"
        })
            print("\n===== Final Battle Report =====")
            print("Winrate:")
            print(f"Teacher_{skill_level}:{(p1win[0]+p1win[1])}/{(ROUNDS*2)} (In Black: {p1win[0]}/{ROUNDS}, In White: {p1win[1]}/{ROUNDS})")
            print(f"Maia_{maia_skill_level}:{(p2win[0]+p2win[1])}/{(ROUNDS*2)} (In Black: {p2win[0]}/{ROUNDS}, In White: {p2win[1]}/{ROUNDS})")
            print("===============================")
    np.save(f'tmp/games_states_history_{output_file_name}.npy', games_states_history, allow_pickle=True)
    with open(f"tmp/games_players_history_{output_file_name}.pkl", "wb") as f:
        pickle.dump(games_players_history, f)
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
