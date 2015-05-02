import sys
from random import sample
from random import shuffle

file1 = sys.argv[1] #Output from generateDataWithLabels.py
label_no = int(sys.argv[2]) #the Question number
outputfile = sys.argv[3] #Output file name

default = 0

ip = open(file1, 'r')
op = open(outputfile, 'w')

range_st = 0 

if label_no == 1:
	range_end = 5018
elif label_no == 2:
	range_end = 1929
elif label_no == 3:
	range_end = 190
else:
	range_end = 665

def splitFile():
	positive_set = []
	negative_set = []
	content = ip.readlines()
	for line in content:
		target_label = int(line.split('\t')[0])
		if target_label == 1:
			positive_set.append(line)
		else:
			negative_set.append(line)
	print(len(positive_set))
	print(len(negative_set))
	return positive_set, negative_set

def undersample(label_no,negative_set):
	dataset = []
	dataset = sample(negative_set, range_end)
	print (len(dataset))
	return dataset


#write the ouput file
def writeOutputToFile(dataset):
	#Read Input File 1 - the data file
	for i,val in enumerate(dataset):
		op.write(val)
	

#main function
sample_set = []
positive_set, negative_set = splitFile()
if label_no == 1:
	sample_set= undersample(label_no, negative_set)
	sample_set = sample_set + positive_set
else:
	sample_set= undersample(label_no, positive_set)
	sample_set = sample_set + negative_set

shuffle(sample_set)
writeOutputToFile(sample_set)
