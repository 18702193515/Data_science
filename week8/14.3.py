from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

newsgroups_data = fetch_20newsgroups(subset='all')
text = newsgroups_data.data

vectorizer = TfidfVectorizer()

vectorized_data = vectorizer.fit_transform(text)
vectorized_array = vectorized_data.toarray()
print(vectorized_array[0])