
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups


categories = ['rec.motorcycles', 'sci.electronics',
              'comp.graphics', 'sci.med']


train_data = fetch_20newsgroups(subset='train',
                                categories=categories, shuffle=True, random_state=42)

print(train_data.target_names)

print("\n".join(train_data.data[0].split("\n")[:3]))
print(train_data.target_names[train_data.target[0]])


for t in train_data.target[:10]:
    print(train_data.target_names[t])



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

knn = KNeighborsClassifier(n_neighbors=7)


clf = knn.fit(X_train_tfidf, train_data.target)


docs_new = ['I have a Harley Davidson and Yamaha.', 'I have a GTX 1050 GPU']

X_new_counts = count_vect.transform(docs_new)

X_new_tfidf = tfidf_transformer.transform(X_new_counts)


predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train_data.target_names[category]))



text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', knn),
])

text_clf.fit(train_data.data, train_data.target)


test_data = fetch_20newsgroups(subset='test',
                               categories=categories, shuffle=True, random_state=42)
docs_test = test_data.data

predicted = text_clf.predict(docs_test)
print('We got an accuracy of',np.mean(predicted == test_data.target)*100, '% over the test data.')
