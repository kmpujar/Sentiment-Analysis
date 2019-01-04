import io
import time
import numpy as np
import operator
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

start_time=time.time()
#List of words to be excluded from stopwords
not_stopwords = {'no', 'not', 'nor'}
#Modified Stop-word List
final_stop_words = set([word for word in text.ENGLISH_STOP_WORDS if word not in not_stopwords])

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
#Function to lemmatize or stem the words
def lemstem(word):
	if word.endswith('e'):
        #Lemmatize the word to adjective form as the sentiment is usually dependant on the adjectives used in the sentence
		words=lemmatizer.lemmatize(word, wn.ADJ)
	else:
		words=ps.stem(word)
	return words

#Read the reviews and corresponding label, pair them and store into a list for further processing
reviews=[]
label=[]
read_file=io.open('reviews.txt','r',encoding='utf-8')
split= [line.strip() for line in read_file]
for line in split:
    reviews.append(line.split('\t')[0])
    label.append(line.split('\t')[1:])
data=list(map(list, zip(reviews, label)))

#Tokenize and replace special characters and numbers
reviews_tokens= [[row.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").replace(";", " ").replace(":", " ").replace("*", " ").replace("(", " ").replace(")", " ").replace("/", " ").
replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").replace("0", "")]
for row in reviews]

#The tokenized words are stemmed to get the root word and stored as processed reviews(p_reviews)
p_reviews =  [" ".join([lemstem(word) for word in sentence[0].split(" ")]) for sentence in reviews_tokens]

#Generating Count Vector
Count_vector = CountVectorizer(stop_words=final_stop_words)
X = Count_vector.fit_transform(p_reviews)
y = np.array(label)
#Initially split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

#Geneating vector using TF-IDF scores
TFIDF_vector = TfidfVectorizer(stop_words=final_stop_words,max_features=1000,binary=False,norm='l2')
X2 = TFIDF_vector.fit_transform(p_reviews)
#Split into train and test  set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.15, random_state=40)



#Using cross validation to select parameters for the SVM classifier
'''
Commented as it takes over 3 hours on a machine with 8gb RAM.

parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.001, 0.10, 0.1, 10, 25, 50, 100, 1000], 'probability':[True, False], 'coef0':[0.0,0.05,0.1]}
#Splitting the training set further into Train and Validation sets for cross validation
cv = ShuffleSplit(n_splits=20, random_state=0, test_size=0.25, train_size=None)
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=cv)
clf.fit(X_train,np.ravel(y_train))
print(clf.best_params_)
print(clf.best_score_ )
print(clf.best_estimator_)

#Output of the above piece of code:
{'kernel': 'rbf', 'C': 10, 'coef0': 0, 'probability': True, 'gamma': 0.01}
0.849861932938856
SVC(C=20, cache_size=200, class_weight=None, coef0=0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''
#Fitting a SVM using the best parameters estimate from cross validation
clf = svm.SVC(C=20, cache_size=200, class_weight=None, coef0=0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
#Test score for Count vector
print("Training accuracy(Count vectorizer)")
print(clf.fit(X_train,np.ravel(y_train)).score(X_train, np.ravel(y_train)))
elap_time=start_time - time.time()
print("Time taken to build classifier is"+str(elap_time))
start_time=time.time()
cvec_score=clf.score(X_test,np.ravel(y_test))
elap_time=start_time - time.time()
print("Time taken to classify test data is"+str(elap_time))
print("Test score for Count Vector: "+"%.4f" %cvec_score)
#Test score for Tf-Idf vector
print("Training accuracy(Tf-Idf)")
print(clf.fit(X2_train,np.ravel(y2_train)).score(X2_train, np.ravel(y2_train)))
tfidf_score=clf.score(X2_test,np.ravel(y2_test))
print("Test score for Tf-Idf Vector: ""%.4f" %tfidf_score)
exit(0)
