#分群：使用 digits 資料集（數字圖片）
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 載入資料集
digits = load_digits()
X = digits.data
y = digits.target

# 建立 KMeans 模型（分類數為 10）
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)

# 顯示部分圖片與其群集標籤
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, img, label in zip(axes.ravel(), digits.images, clusters):
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Cluster {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()
