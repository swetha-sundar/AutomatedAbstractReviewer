#!/usr/bin/python

from xml.etree.ElementTree import ElementTree
import xml.etree.ElementTree as ET
import sys


TITLE = "title"
KEYWORDS = "keywords"
ABSTRACT = "abstract"
NONE = "none"

def writeToFile(records, op):
	for rec_id in records.iterkeys():
		#print rec_id
		op.write(rec_id)
		op.write("\t")
		for elements in records[rec_id]:
			for key, val in elements.items():
				op.write(val)
				op.write("\t")
		op.write("\n")

tofind_record = 'records/record'
tofind_title = 'titles/title'
tofind_keyword = 'keywords/keyword'
tofind_abstractid = 'rec-number'
tofind_abstract = 'abstract'


filename = sys.argv[1]
outputfile = sys.argv[2]

op = open(outputfile,'w')

#read the xml file
doc = ET.parse(filename)

#obtain the root
root = doc.getroot() 

#get all elements of interest
for record in root.iterfind(tofind_record):
	
	record_dict = {}
	abstractElements = []
	title_dict = {}
	keywords_dict = {}
	abstract_dict = {}
	keywords = []
	
	#Get the abstract id
	rec_num = record.find(tofind_abstractid)
	abs_id = rec_num.text
#	print abs_id
	
	#Get the title
	for title in record.iterfind(tofind_title):
		title_dict[TITLE] = title.text
		abstractElements.append(title_dict)
#		print abstractElements
	
	#Get the keywords
	for keyword in record.iterfind(tofind_keyword):
		keywords.append(keyword.text)
	keywords_str = ",".join(keywords)
	keywords_dict[KEYWORDS] = keywords_str
	abstractElements.append(keywords_dict)
	
	#Get the abstract
	abstract = record.find(tofind_abstract)
	if abstract is None:
		abstract = NONE
		abstract_dict[ABSTRACT] = abstract
	else:
		abstract_dict[ABSTRACT] = abstract.text
	
	abstractElements.append(abstract_dict)

	#create the entire record
	record_dict[abs_id] = abstractElements
	writeToFile(record_dict, op)
op.close()
#print record_dict
