from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

newsgroups_data = fetch_20newsgroups(subset='all')
text = newsgroups_data.data
labels = newsgroups_data.target

vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(text)

X_train, X_test, y_train, y_test = train_test_split(vectorized_data, labels, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("训练集分类准确率：", train_accuracy)
print("测试集分类准确率：", test_accuracy)