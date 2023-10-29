import numpy as np

def f(x):
    return 9*x+8

x=np.linspace(0,3,30)
y=f(x)
noise=np.random.normal(0,1,30)
noised_y=y+noise
# print(noise)
# print(noised_y)

from sklearn.linear_model import LinearRegression
x = np.array(x).reshape(-1,1)
lr = LinearRegression()
lr.fit(x, noised_y)
print(lr.coef_)
print(lr.intercept_)