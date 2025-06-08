import numpy as np
from scipy.optimize import minimize

# 七段顯示器真值表（7 bits 輸入）
seven_segment_truth_table = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}

# 對應數字的二進位輸出（4 bits）
binary_outputs = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 0, 1, 1),
    4: (0, 1, 0, 0),
    5: (0, 1, 0, 1),
    6: (0, 1, 1, 0),
    7: (0, 1, 1, 1),
    8: (1, 0, 0, 0),
    9: (1, 0, 0, 1),
}

# 建立訓練資料
input_vectors = np.array([seven_segment_truth_table[i] for i in range(10)])  # shape: (10, 7)
target_outputs = np.array([binary_outputs[i] for i in range(10)])  # shape: (10, 4)

# 損失函數 (均方誤差)
def loss_function(w):
    w = np.array(w).reshape(7, 4)
    predictions = input_vectors @ w
    return np.mean((predictions - target_outputs) ** 2)

# 初始權重（隨機）
initial_weights = np.random.rand(7 * 4)

# 執行優化
result = minimize(loss_function, initial_weights, method='BFGS')
trained_weights = result.x.reshape(7, 4)

# 預測函數
def predict(segment_input):
    raw_output = segment_input @ trained_weights
    binary_output = (raw_output >= 0.5).astype(int)  # 將結果轉為 0/1
    return binary_output

# 測試預測
print("=== 預測結果 ===")
for num, segment in seven_segment_truth_table.items():
    binary_prediction = predict(np.array(segment))
    binary_str = ''.join(map(str, binary_prediction))
    print(f"{num}: {segment} -> 預測 = {binary_str} (應為 {''.join(map(str, binary_outputs[num]))})")
