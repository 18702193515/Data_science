from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

newsgroups_data = fetch_20newsgroups(subset='all')

num_samples = 10 
text_data = newsgroups_data.data[:num_samples]

vectorizer = CountVectorizer()

vectorized_data = vectorizer.fit_transform(text_data)
vectorized_array = vectorized_data.toarray()
print(vectorized_array)