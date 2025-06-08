from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# 載入乳癌資料集（非 iris、非 wine）
X, y = load_breast_cancer(return_X_y=True)

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 Gradient Boosting 模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
