#回歸：使用 diabetes 糖尿病資料集
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 載入資料集
X, y = load_diabetes(return_X_y=True)

# 切分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("均方誤差 (MSE):", mse)
