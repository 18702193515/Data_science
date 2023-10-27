import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import euclidean

iris = load_iris()

X = iris.data
y = iris.target

labels = np.unique(y) 
centers = []
for label in labels:
    X_label = X[y == label] 
    center = np.mean(X_label, axis=0)
    centers.append(center)

distances = []
for i, center in enumerate(centers):
    X_label = X[y == labels[i]]  
    distance = [euclidean(x, center) for x in X_label]  
    distances.extend(distance)

print("数据点到中心点的欧氏距离:")
for i, distance in enumerate(distances):
    label = iris.target_names[y[i]]
    print("数据点", i+1, "属于类别", label, "，距离为:", distance)