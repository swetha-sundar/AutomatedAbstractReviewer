from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

import numpy as np
import sys
import matplotlib.pyplot as plot

if (len(sys.argv) != 7):
	print "Invalid #parameters. \n 4 Arguments necessary \n\t corpus_name \n\t gold_labels \n\t question_num \n\t classifier_name Options are: sgd/linearsvc/svc/nb/nearest_centroid \n\t output file name\n"
	sys.exit()

train_file = sys.argv[1] #output of under_sampling or generateDataWithLabels.py
test_file = sys.argv[2] 
gold_labels = sys.argv[3] #relevant.txt file
label_num = sys.argv[4]  #the question number
cls = sys.argv[5] #classifier option
op_file = sys.argv[6] #output file name

op = open(op_file, 'w')

if (cls not in ['sgd', 'linearsvc', 'nb', 'nearest_centroid', 'svc']):
	print "Wrong classifier option. Options are: sgd/linearsvc/svc/nb/nearest_centroid\n"
	sys.exit()

#import train/test data
def fetch_corpus(file_name):
	f = open(file_name, 'r')
	content = f.readlines()
	return content

#remove Labels from data
def extractElements(data):
	labels = []
	absids = []
	data_absids = []
	only_data = []
	for line in data:
		labels.append(int(line.split("\t")[0]))
		absids.append(line.split("\t")[1])
		data_absids.append(line.split("\t",1)[1])
		only_data.append(line.split("\t",2)[2])
	return labels, absids,data_absids,only_data


#Comparison against the gold standard
def checkRelevance(abs_id):
	with open(gold_labels) as f:
		content = f.readlines()
		for line in content:
			if (line.strip() == abs_id):
				return True
		return False

#Building y_test
def getTestLabels(test_data):
	y_test = []
	for line in test_data:
		abs_id = line.split('\t',1)[0]
		isRelevant = checkRelevance(abs_id)
		if (isRelevant):
			y_test.append(int(1))
		else:
			y_test.append(int(0))
	return y_test

def convertToList(data):
	new_data = []
	prevx = 0
	s = ""
	
	for (x,y), value in np.ndenumerate(data.tolist()):
		if (prevx != x):
			new_data.append(s)
			s = ""
		s =  s + value + "\t"
		prevx = x		
	
	new_data.append(s)
	return new_data

def convertToNum(data):
	new_data = []
	for i,val in enumerate(data):
		new_data.append(int(val))
	return new_data

#classifier function. Pass any classifier as a parameter. Not currently using it. 
def classify(clf):
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	score = metrics.f1_score(y_test, pred,pos_label=1)
	accuracy = metrics.accuracy_score(y_test, pred)
	print "accuracy:"  
	print accuracy
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)	
	print fpr
	auc_val =  metrics.auc(fpr, tpr)
	print "AUC:" 
	print auc_val
	print("classification report:")
	print(metrics.classification_report(y_test, pred))
	clf_descr = str(clf).split('(')[0]
	
	return clf_descr, score

#performs grid search
def performGridSearch(pipeline, parameters, train_data, train_target, test_data, test_target):
	print "Performing Grid Search.....\n"

	grid_search = GridSearchCV(pipeline,parameters,cv=5, n_jobs=-1,verbose=1,scoring='f1') 
	model = grid_search.fit(train_data,train_target)
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_params = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t %s: %r" % (param_name, best_params[param_name]))
	print "Predicting..."
	y_pred = model.predict(test_data)
	print(metrics.classification_report(test_target,y_pred))
	fpr, tpr, thresholds = metrics.roc_curve(test_target, y_pred, pos_label=1)	
	print fpr
	auc_val =  metrics.auc(fpr, tpr)
	print "AUC:" 
	print auc_val
	return y_pred

def writePredToFile(predicted_data, absids):
	for i in range(len(absids)):
		op.write(str(absids[i]))
		op.write("\t")
		op.write(str(predicted_data[i]))
		op.write("\n")
#main method

#1. obtain the train and test split
print "Splitting the data into train and test.....\n"
train = fetch_corpus(train_file)
test = fetch_corpus(test_file)
print "Train and Test subsets created!!! \n"

#2. remove the labels from the test set.
print "Preprocessing....\n"

#3. remove the abstract ids and the labels from the training & testing set (no special meaning to have it in the test set)
#4.get just the labels of the training data and convert them to integers (from string)

train_labels,train_absids,train_data_absids,train_data = extractElements(train)
test_labels,test_absids,test_data_absids,test_data = extractElements(test)

#5. except for question 1, test labels for all other questions are crowd-source labels
if (label_num == '1'):
	test_labels = getTestLabels(test_data_absids)
print len(test_labels)
print "Preprocessing complete!!\n"

#SGD CLASSIFIER
if (cls.lower() == 'sgd'):
	pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
		('clf',SGDClassifier(alpha = 0.0001, n_iter = 10, penalty = 'elasticnet' ))])

	parameters = {
			'vect__max_df':(0.5,0.75,1.0),
			'vect__max_features':(10000,20000,50000),
			'vect__ngram_range':((1,1),(1,2)),
			'tfidf__use_idf':(True,False),
			#'clf__class_weight':(0.01, 100)
			'clf__alpha':(0.001, 0.01,100),
			#'clf__penalty':('l2','elasticnet'),
			#'clf__n_iter':(10,50,80)
		     }
elif (cls.lower() == 'linearsvc'):
#LINEAR SVC

	pipeline = Pipeline([('vect', CountVectorizer(max_features=20000, ngram_range=(1,2))), ('tfidf', TfidfTransformer(use_idf=True)), 
		('clf',LinearSVC(C=0.1))])
	
	parameters = {
			'vect__max_df':(0.5,0.75,1.0),
			#'vect__max_features':(10000,20000,50000),
			#'vect__ngram_range':((1,1),(1,2)),
			#'tfidf__use_idf':(True,False),
			#'clf__C':(0.01, 0.1, 1),
        	     }
elif (cls.lower() == 'linearsvr'):
#LINEAR SVR
	pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
		('clf',LinearSVR())])

	parameters = {
			'vect__max_df':(0.75,1.0),
			'vect__max_features':(10000,20000,50000),
			'vect__ngram_range':((1,1),(1,2)),
			'tfidf__use_idf':(True,False),
			'clf__C':(0.01, 0.1, 1, 100),
			'clf__max_iter':(10,50,80)
		     }

elif (cls.lower() == 'svc'):
#SVC
	pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
		('clf',SVC())])
	
	parameters = {
			'vect__max_df':(0.5,0.75,1.0),
			'vect__max_features':(10000,20000,50000),
			'vect__ngram_range':((1,1),(1,2)),
			'tfidf__use_idf':(True,False),
			'clf__C':(0.01, 0.1, 1),
			'clf__kernel':('rbf', 'linear')
		     }
elif (cls.lower() == 'nb'):
#NB
	
	pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
		('clf',GaussianNB())])
	
	parameters = {
			'vect__max_df':(0.5,0.75,1.0),
			'vect__max_features':(10000,20000,50000),
			'vect__ngram_range':((1,1),(1,2)),
			'tfidf__use_idf':(True,False)
		     }
	y_pred = performGridSearch(pipeline, parameters, train_data, train_labels, test_data, test_labels)
 		
elif (cls.lower() == 'nearest_centroid'):
#Nearest Centroid
	
	pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
		('clf',NearestCentroid())])
	
	parameters = {
			'vect__max_df':(0.5,0.75,1.0),
			'vect__max_features':(10000,20000,50000),
			'vect__ngram_range':((1,1),(1,2)),
			'tfidf__use_idf':(True,False),
			'clf__metric':('euclidean','l1','l2')
		     }
else:
	print "Options are: sgd/linearsvc/svc/nb/nearest_centroid"
	sys.exit()

if (cls.lower() != 'nb'):
	y_pred = performGridSearch(pipeline, parameters, train_data, train_labels, test_data, test_labels)

print "Writing predicted values to file...\n"
writePredToFile(y_pred, test_absids)
print "Writing Complete!!!"
'''
indices = np.arange(len(results))
results = [[x[i] for x in results] for i in range(2)]
clf_names, score = results

plot.figure(figsize=(12,8))
plot.title("Score")
plot.barh(indices, score, 0.2, label="score", color='r')
plot.yticks(())
plot.legend(loc='best')
plot.subplots_adjust(left=0.25)
plot.subplots_adjust(top=0.95)
plot.subplots_adjust(bottom=0.05)
plot.show()
'''

