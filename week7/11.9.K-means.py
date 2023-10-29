import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
iris = load_iris()
X = iris.data

# 创建KMeans模型并进行聚类
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(X)

labels = kmeans.labels_

for i, label in enumerate(labels):
    print("数据点", i+1, "的类别标签:", label)