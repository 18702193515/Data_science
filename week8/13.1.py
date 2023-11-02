import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data  
y = iris.target  

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()