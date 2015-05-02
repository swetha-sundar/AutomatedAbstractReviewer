import sys
import csv

file1 = sys.argv[1] #Output from preprocess.py
file2 = sys.argv[2] #The Raw Turk Results file
label_no = sys.argv[3] #the Question number
outputfile = sys.argv[4] #Output file name

QUESTION = "Question"
ABSTRACTID = "AbstractId"
THRESHOLD = 20
LABEL_VALUES = {"NOINFO", "CANTTELL"}
default = "0"

op = open(outputfile, 'w')

def getLabelsQ4(label_no):
	with open(file2, 'r') as ip2:
		content2 = csv.DictReader(ip2)
		labels = {}
		i = 0
		for row in content2:
			absid = row[ABSTRACTID]
			field = QUESTION+label_no
			label_val = row[field].upper()
			if (label_val == "-"):
				continue
			elif ((label_val in LABEL_VALUES) or ((int(label_val)) >= THRESHOLD)):
				target_label = "YES"
			else:
				target_label = "NO"
			
			if not absid in labels:
				labels[absid] = [target_label]
			else:
				labels[absid].append(target_label)
		return labels

#Read the Turk Results - to get the labels per question
def getTargetLabels(label_no):
	labels = {}
	if (label_no == '4'):
		labels = getLabelsQ4(label_no)
		return labels

	with open(file2, 'r') as ip2:
		content2 = csv.DictReader(ip2)
		i = 0
		for row in content2:
			absid = row[ABSTRACTID]
			field = QUESTION+label_no
			label_val = row[field].upper()
			if (label_val == '-'):
				continue
			elif (label_val == "NO"):
				target_label = "NO"
			else:
				target_label = "YES"
			if not absid in labels:
				labels[absid] = [target_label]
			else:
				labels[absid].append(target_label)
		return labels

#max votes
def doMaxVoting(labels):
	final_labels = {}
	for key in labels.iterkeys():
		no_count = 0
		yes_count = 0
		for val in labels[key]:
			if (val.upper() == 'YES'):
				yes_count = yes_count + 1
			else:
				no_count += 1
		if(yes_count >= no_count):
			final_labels[key] = int(1)
		else:
			final_labels[key] = int(0)
	return final_labels

#repetitive additions
def repeatRecords(labels):
	final_labels = {}
	for key in labels.iterkeys():
		for val in labels[key]:
			if (val.upper() == 'YES'):
				ans = 1
			else:
				ans = 0
			if not key in final_labels:
				final_labels[key] = [ans]
			else:
				final_labels[key].append(ans)
	return final_labels

'''
#Comparison against the gold standard
def checkRelevance(abs_id):
	with open(file2) as ip2:
		content2 = ip2.readlines()
		for line in content2:
			if (line.strip() == abs_id):
				return True
		return False
'''
#write the ouput file
def writeOutputToFile(final_labels):
	#Read Input File 1 - the data file
	not_found = []
	ip1 = open(file1, 'r')
	for line in ip1:
		op_line = line.split('\t',1)[1]
		absid = line.split('\t',1)[0]
		if (absid in final_labels): 
			for value in final_labels.get(absid):
				#print absid
				#label = final_labels.get(absid) #Abstract ID 996 not found in crowd results
				op.write(str(value))
				op.write("\t")
				op.write(line)
		else:
			not_found.append(absid)
	#print "Abstract Ids not found:"
	#print not_found
#main function
target_labels = {}
temp_labels = {}
temp_labels = getTargetLabels(label_no)
#target_labels =doMaxVoting(temp_labels)
target_labels = repeatRecords(temp_labels)
writeOutputToFile(target_labels)

'''	
for line in content1:
	temp = line.split('\t',1)[0]
	abs_id = temp.split(':',1)[1]
	isRelevant = checkRelevance(abs_id)
	if (isRelevant):
		writeLineToOutput(line)
'''
