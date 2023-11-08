from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)

print("CountVectorizer Accuracy:", nb_count.score(X_test_count, y_test))

print("\nCountVectorizer Classification Report:")
print(classification_report(y_test, nb_count.predict(X_test_count), target_names=data.target_names))