#分類：使用 wine 資料集
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 載入資料集
X, y = load_wine(return_X_y=True)

# 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立模型
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 預測與評估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
