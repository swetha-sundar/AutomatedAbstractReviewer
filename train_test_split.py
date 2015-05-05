from sklearn import cross_validation
import sys
import numpy as np

TRAIN_PERCENT=0.6
TEST_PERCENT=0.4

ip = sys.argv[1] #input file. (preprocessed data)
op1 = sys.argv[2] #file to write the train data
op2 = sys.argv[3] #file to write the test data

f = open(ip,'r')
o1 = open(op1, 'w')
o2 = open(op2, 'w')

def writeToFile(data,op):
	for i in range(len(data)):
		op.write(data[i])
		op.write("\n")
	op.close()

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

data = np.array([]).reshape(0,)
corpus = []
content = f.readlines()
for line in content:
	new_line = line.strip()
	elements = new_line.split('\t')
	corpus.append(elements)
data = np.array(corpus)
train, test = cross_validation.train_test_split(data,train_size=TRAIN_PERCENT, test_size=TEST_PERCENT)

train_list = convertToList(train)
test_list = convertToList(test)

writeToFile(train_list,o1)
writeToFile(test_list,o2)
