import io
import time
import random
import operator
import numpy as np
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = time.time()
#Set of stop words in NLTK corpus
stop_words = set(stopwords.words('english'))
#List of words to be excluded from stopword list
not_stopwords = {'no', 'not', 'nor'}
#Final list of stop word
final_stop_words = set([word for word in stop_words if word not in not_stopwords])

reviews=[]
label=[]

#Read the reviews anc corresponding labels to lists
read_file=io.open('reviews.txt','r',encoding='utf-8')
split= [line.strip() for line in read_file]
for line in split:
    reviews.append(line.split('\t')[0])
    label.append(line.split('\t')[1:])
data=[list(i) for i in zip(reviews,label)]

#Preprocess the documents by eliminating numbers and special characters
reviews_c= [[row[0].lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").replace(";", " ").replace(":", " ").replace("*", " ").replace("(", " ").replace(")", " ").replace("/", " ").
replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").replace("0", ""),row[1]]
for row in data]

#Tokenize the documents
token_review = [[obs[0].split(), list(map(int, (obs[1])))] for obs in reviews_c]

#Eliminate stop words from the tokenized reviews
for doc in token_review:
	doc[0]=[word for word in doc[0] if word not in list(final_stop_words)]

#Random shuffling of dataset before splitting to Train and test sets
random.Random(47).shuffle(token_review)

#Split into train and test in the ratio 80:20
slice_index=80*3000/100
train_data=list(token_review[:int(slice_index)])
test_data=list(token_review[int(slice_index):])

#Split the test data into positive and negative for further probability calculations
train_pos=[]
train_neg=[]
for doc in train_data:
    if(doc[1][0]==1):
        train_pos.append(doc)
    else:
        train_neg.append(doc)

#Generate a list of all the words present in the corpus
vocab_list=[]
for doc in train_data:
	for word in doc[0]:
		if word in vocab_list:
			continue
		else:
			vocab_list.append(word)

#Calculate logprior for Positive class : log((no. of docs in positive class)/ total num of docs
logprior_pos=np.log((len(train_pos)+1))-np.log(len(train_data))
#Calculate logprior for Negative class :  log((no. of docs in negative class)/ total num of docs
logprior_neg=np.log((len(train_neg)+1))-np.log(len(train_data))

#num of words in negative class
count_neg=0
for d in train_neg:
	count_neg+=len(d[0])
total_neg=count_neg+len(vocab_list)

#num of words in positive class
count_pos=0
for d in train_pos:
	count_pos+=len(d[0])
total_pos=count_pos+len(vocab_list)


#computing the probability of occurrance of word given negative class
dic_neg={}
loglikelihood=[None]*2
for wi in vocab_list:
            #Count number of times wi appears in documents with negative label
            count_wi_in_D_c = 0
            for d in train_neg:
                for word in d[0]:
                    if word == wi:
                        count_wi_in_D_c = count_wi_in_D_c + 1
            numer = count_wi_in_D_c + 1
            dic_neg[wi] = np.log(numer)-np.log(total_neg)
loglikelihood[0] = dic_neg

dic_pos={}
for wi in vocab_list:
            #Count number of times wi appears in documents with positive label
            count_wi_in_D_c = 0
            for d in train_pos:
                for word in d[0]:
                    if word == wi:
                        count_wi_in_D_c = count_wi_in_D_c + 1
            numer = count_wi_in_D_c + 1
            dic_pos[wi] = np.log(numer)-np.log(total_pos)
loglikelihood[1] = dic_pos

logprior=[logprior_neg,logprior_pos]
logpost = [None] * 2

elapsed_time = time.time() - start_time
print("Time taken to build the model(in secs):"+str(elapsed_time))
#print(elapsed_time)
#Computes probability of each class given a document
start_time = time.time()
counterrr=0
logpost2=[None]*2
#Predicts the class for each documet in the test set
for doc in test_data:
	for i in range(0,2):
		sumloglihood=0
		for word in doc[0]:
			if word in vocab_list:
				sumloglihood+=loglikelihood[i][word]
		logpost[i] = logprior[i] + sumloglihood

#The class with higher probability is chosen as the prediction
	if(logpost.index(max(logpost))==doc[1][0]):
		counterrr= counterrr+1
elapsed_time = time.time() - start_time
print("Time taken to classify test data(in secs):"+str(elapsed_time))
prediction=counterrr/float(len(test_data))
print("Accuracy of NB classifier is %.4f"%prediction)

''' Comparision with Naive Bayes implementation using SCi-Kit learn'''
p_reviews=[]
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

""" function to stem the words in the documents """
def lemstem(word):
	if word.endswith('e'):
		words=lemmatizer.lemmatize(word)
	else:
		words=ps.stem(word)
	return words

""" To lemmatize and stem to root word """
p_reviews =  [" ".join([lemstem(word) for word in sentence.split(" ")]) for sentence in reviews]

y = np.array(label)
clf = MultinomialNB()

"""Naive Bayes classification using Count Vector"""
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(p_reviews)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train,np.ravel(y_train))
print("Accuracy of NB using Sci-Kit with Cout vectorizer is")
print(clf.score(X_test,y_test,sample_weight=None))

"""Naive Bayes classification using Tf-Idf vector"""
vectorizer2 = TfidfVectorizer(stop_words='english',max_features=1000,binary=False,norm='l2')
X2 = vectorizer2.fit_transform(p_reviews)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
clf.fit(X2_train,np.ravel(y_train))
print("Accuracy of NB using Sci-Kit with Tf-Idf vectorizer is")
print(clf.score(X2_test,y_test,sample_weight=None))

exit(0)
