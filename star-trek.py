# Write-up: https://ology.github.io/2020/03/03/predicting-star-trek-characters-with-naive-bayes/

# LOAD FILES INTO A DATAFRAME
import pandas as pd
import nltk.data

def file_to_df(path, name):
    fh = open(path + '/' + name + '.txt')
    data = fh.read()
    fh.close()
    tokenizer = nltk.data.load('nltk_data/tokenizers/punkt/english.pickle')
    return pd.DataFrame(
        zip(
            tokenizer.tokenize(data),
            ([name] * len(tokenizer.tokenize(data)))
        ),
        columns=['text','person']
    )

path = 'Kirk-Spock-McCoy'

mccoy = file_to_df(path, 'mccoy')
spock = file_to_df(path, 'spock')
kirk  = file_to_df(path, 'kirk')

result = pd.concat([mccoy, spock, kirk])


# SHOW COUNTS
import matplotlib.pyplot as plt
result.groupby('person').text.count().plot.bar(ylim=0)
plt.show()


# GET THE TRAIN/TEST DATA
X = result.text
y = result.person

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# LEARN THE VOCABULARY
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer() #stop_words='english' => 0.6598099205832574

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
X_dtm = vect.transform(X)


# TF-IDF SCALE THE VOCABULARY
from sklearn.feature_extraction.text import TfidfTransformer
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_dtm)
#X_test_tfidf = tfidf_transformer.fit_transform(X_test_dtm)


# TRAIN A CLASSIFIER
from sklearn.naive_bayes import MultinomialNB

#clf = MultinomialNB().fit(X_train_tfidf, y_train)
#y_pred = clf.predict(X_test_tfidf) # 0.628563989063924

clf = MultinomialNB().fit(X_train_dtm, y_train)
y_pred = clf.predict(X_test_dtm) # 0.6656685327431324


# EVALUATE THE MODEL ACCURACY
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
#metrics.confusion_matrix(y_test, y_pred)


# PREDICT ARBITRARY PHRASES
docs = [ "Captian's log, Stardate...", 'That is highly illogical.', "He's dead Jim." ]

X_new_counts = vect.transform(docs)
predicted = clf.predict(X_new_counts)

for doc, who in zip(docs, predicted):
    print('%r => %s' % (doc, who))


# MOST PREDICTIVE TOKENS
vect = CountVectorizer(stop_words='english')
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_dtm)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_dtm)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_train_tokens = vect.get_feature_names()

kirk_token_count  = clf.feature_count_[0, :]
mccoy_token_count = clf.feature_count_[1, :]
spock_token_count = clf.feature_count_[2, :]

tokens = pd.DataFrame(
    {
        'token': X_train_tokens,
        'mccoy': mccoy_token_count,
        'spock': spock_token_count,
        'kirk': kirk_token_count
    }
).set_index('token')
tokens.head()

tokens['kirk']  = tokens.kirk + 1
tokens['mccoy'] = tokens.mccoy + 1
tokens['spock'] = tokens.spock + 1

tokens['kirk']  = tokens.kirk / clf.class_count_[0]
tokens['mccoy'] = tokens.mccoy / clf.class_count_[1]
tokens['spock'] = tokens.spock / clf.class_count_[2]

tokens.sort_values('kirk', ascending=False).head(10)
tokens.sort_values('mccoy', ascending=False).head(10)
tokens.sort_values('spock', ascending=False).head(10)


# Some more phrases!
def who_said(docs):
    X_new_counts = vect.transform(docs)
    predicted = clf.predict(X_new_counts)
    for doc, who in zip(docs, predicted):
        print('%r => %s' % (doc, who))

docs = [ 'take me to the doctor, captain', 'jim, where is spock?' ]
who_said(docs)
# 'take me to the doctor, captain' => spock
# 'jim, where is spock?' => mccoy

docs = [ 'our father who art in heaven' ]
who_said(docs)
# 'our father who art in heaven' => kirk


# SVM
#from sklearn.linear_model import SGDClassifier
#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
#clf.fit(X_train_tfidf, y_train)
#y_pred = clf.predict(X_test_tfidf)
#metrics.accuracy_score(y_test, y_pred) # 0.6165863819815128


# GRIDSEARCH
from sklearn.model_selection import GridSearchCV

# prepare a range of values to test
parameters = {'alpha': [0.01,0.1,1,1.5,2]}

# create and fit a ridge regression model, testing each alpha
model = MultinomialNB()
grid = GridSearchCV(estimator=model, param_grid=parameters)
grid.fit(X_train_dtm, y_train)

# summarize the results of the grid search
print(grid.best_score_) # 0.661299422768109
print(grid.best_estimator_.alpha) # 1

grid.fit(X_dtm, y) # 0.6498925851181564 - lower??

