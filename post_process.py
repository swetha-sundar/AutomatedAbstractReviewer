import sys

ip1 = open(sys.argv[1],'r')
ip2 = open(sys.argv[2],'r')
ip3 = open(sys.argv[3],'r')
ip4 = open(sys.argv[4],'r')

def readFile(file_ptr):
	f = file_ptr.readlines()
	return f 

def parseIntoMaps(f):
	predValDict = {}
	for line in f:
		absid = line.split("\t")[0]
		val = line.split("\t")[1]
		predValDict[0] = val
	return predValDict

def predict(dict1, dict2, dict3, dict4):
	pred = {}
	tcnt = 0
	fcnt = 0
	for key in dict1:
		if (key in dict2 and key in dict3 and key in dict4):
			tcnt = tcnt + 1
			if dict1[key] == "1" and dict2[key] == "1" and dict3[key] == "1" and dict4[key] == "1":
				pred[key] = 1
			else:
				pred[key] = 0
		else:
			#pred[key] = 0
			fcnt = fcnt + 1

	print "tcnt:" 
	print tcnt
	print "fcnt:"+fcnt+"\n"
	print len(dict1)
	print len(pred)
	return pred

#main method

#1. read the content of all files
f1 = readFile(ip1)
f2 = readFile(ip2)
f3 = readFile(ip3)
f4 = readFile(ip4)

#2. store the contents in hashmaps
dict1 = parseIntoMaps(f1)
dict2 = parseIntoMaps(f2)
dict3 = parseIntoMaps(f3)
dict4 = parseIntoMaps(f4)

#3.predict the final output based on test set in dictionary 1 
pred = predict(dict1,dict2,dict3,dict4)

#4. TODO: check relevance with gold labels
