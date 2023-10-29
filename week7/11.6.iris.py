from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x, y = shuffle(x, y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

predictions = logreg.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("准确率：", accuracy)