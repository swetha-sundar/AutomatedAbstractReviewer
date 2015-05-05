import sys
from sklearn import metrics

ip1 = open(sys.argv[1],'r')
ip2 = open(sys.argv[2],'r')
ip3 = open(sys.argv[3],'r')
ip4 = open(sys.argv[4],'r')

expert_label = open(sys.argv[5],'r')

def readFile(file_ptr):
	f = file_ptr.readlines()
	return f 

def parseIntoMaps(f):
	predValDict = {}
	for line in f:
		absid = line.split("\t")[0]
		val = line.split("\t")[1]
		predValDict[absid] = int(val)
	return predValDict

def parseIntoMap(f):
	goldDict = {}
	for line in f:
		absid = line.strip()
		goldDict[absid] = 1
	return goldDict

def predict(dict1, dict2, dict3, dict4):
	pred = {}
	tcnt = 0
	fcnt = 0
	val = 0
	for key in dict1:
		val = dict1[key]
		if (key in dict2 and key in dict3 and key in dict4):
			tcnt = tcnt + 1
			if dict1[key] == 1 and dict2[key] == 1 and dict3[key] == 1 and dict4[key] == 1:
				pred[key] = 1
			else:
				val = dict1[key] + dict2[key] + dict3[key] + dict4[key]
				if (val > 1):
					pred[key] = 1
				else:
					pred[key] = 0
		else:
			if (key in dict2):
				val += dict2[key]
			if (key in dict3):
				val += dict3[key]
			if (key in dict4):
				val += dict4[key]
			if (val > 1):
				pred[key] = 1
			else:
				pred[key] = 0
			fcnt = fcnt + 1
	print "tcnt:" 
	print tcnt
	print "fcnt:"
	print fcnt
	return pred
def populateTargetLabel(goldDict, dict1):
	for key in dict1:
		if key in goldDict:
			continue
		else:
			goldDict[key] = 0
	return goldDict

def checkRelevance(predDict, goldDict):
	tpos=0 
	tneg=0
	fpos=0
	fneg=0
	for key in predDict:
		if key in goldDict:
			if goldDict[key] == 1 and predDict[key] == 1:
				tpos += 1
			elif goldDict[key] == 1 and predDict[key] == 0:
				fneg += 1
			elif goldDict[key] == 0 and predDict[key] == 0:
				tneg += 1
			else:
				fpos += 1
	
	print "TruePos\tTrueNeg\tFalsePos\tFalseNeg\n"
	print tpos,"\t",tneg,"\t",fpos,"\t",fneg
	recall = float(tpos/float(tpos + fneg))
	print "Recall:",recall
	precision = float(tpos/float(tpos + fpos))
	print "Precision:",precision
	#fpr = fpos/(fpos + tneg)
	#auc = metrics.auc(fpr,recall)
	#print "AUC:",auc
	
#main method

#1. read the content of all files
f1 = readFile(ip1)
f2 = readFile(ip2)
f3 = readFile(ip3)
f4 = readFile(ip4)

gold_labels = readFile(expert_label)
#print gold_labels
#2. store the contents in hashmaps
dict1 = parseIntoMaps(f1)
dict2 = parseIntoMaps(f2)
dict3 = parseIntoMaps(f3)
dict4 = parseIntoMaps(f4)

gold_labels_dict = parseIntoMap(gold_labels)
gold_pred_dict = populateTargetLabel(gold_labels_dict, dict1)
#3.predict the final output based on test set in dictionary 1 
pred = predict(dict1,dict2,dict3,dict4)

#4. TODO: check relevance with gold labels
final = checkRelevance(pred, gold_pred_dict)

#5. WriteToFile
#writeToFile(final)
