import pandas as pd
import os

def correct_maia_skill_level(input_path, output_path):
    """
    讀取 CSV 檔案，並根據預定的映射關係修正 'Maia_skill_level' 欄位。

    Args:
        input_path (str): 輸入的原始 CSV 檔案路徑。
        output_path (str): 修正後要輸出的 CSV 檔案路徑。
    """
    # --- 步驟 1: 定義錯誤值與正確值的對應關係 ---
    # 這些是您提供的列表
    incorrect_levels = ['500', '1000', '2500', '3000', '3500', '4000', '5000']
    correct_levels = ['650', '800', '1000', '1200', '1550', '1650', '1750']
    
    # 建立一個映射字典，用於替換。
    # 由於 CSV 中的 skill level 可能是數字 (int)，我們將 key 轉換為整數以確保能成功匹配。
    skill_level_map = {int(old): int(new) for old, new in zip(incorrect_levels, correct_levels)}
    
    print("將使用以下規則進行替換：")
    print(skill_level_map)
    print("-" * 30)

    try:
        # --- 步驟 2: 讀取 CSV 檔案 ---
        print(f"正在讀取檔案: {input_path}...")
        df = pd.read_csv(input_path)
        
        # 記錄原始的 skill levels 以供後續檢查
        original_skill_levels = df['Maia_skill_level'].unique()

        # --- 步驟 3: 使用 .replace() 方法進行替換 ---
        print("正在替換 'Maia_skill_level' 欄位中的值...")
        df['Maia_skill_level'] = df['Maia_skill_level'].replace(skill_level_map)
        
        # --- 步驟 4: 檢查是否有未被替換的值 ---
        unmapped_values = [level for level in original_skill_levels if level in skill_level_map.keys()]
        if len(unmapped_values) != len(original_skill_levels):
             print("\n⚠️ 警告: CSV 中存在一些 'Maia_skill_level' 未在替換規則中，這些值將被保留原樣。")
             
        # --- 步驟 5: 將結果儲存到新的 CSV 檔案 ---
        # index=False 表示不要將 DataFrame 的索引寫入檔案中
        df.to_csv(output_path, index=False)
        
        print("\n✅ 清理完成！")
        print(f"已處理 {len(df)} 筆記錄。")
        print(f"結果已儲存至: {output_path}")

    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 '{input_path}'。請確認檔案路徑是否正確。")
    except KeyError:
        print(f"❌ 錯誤：CSV 檔案中缺少 'Maia_skill_level' 欄位，請檢查欄位名稱是否正確。")
    except Exception as e:
        print(f"❌ 發生了未知的錯誤: {e}")


# ===============================================================
# --- 您需要修改的地方 ---
# 1. 設定您的來源檔案路徑
# 例如: 'C:/Users/YourUser/Desktop/my_data.csv'
input_file = './results/TvT-400.csv' 

# 2. 設定您希望儲存的目標檔案路徑
# 例如: 'C:/Users/YourUser/Desktop/my_data_corrected.csv'
output_file = './results/TvT-400-new.csv' 
# ===============================================================


# --- 執行腳本 ---
if __name__ == '__main__':
    # 確保您已經修改了上面的 input_file 和 output_file 路徑
    if 'path/to/your' in input_file or 'path/to/your' in output_file:
        print("!!! 請先在程式碼中修改 'input_file' 和 'output_file' 的路徑 !!!")
    else:
        correct_maia_skill_level(input_file, output_file)