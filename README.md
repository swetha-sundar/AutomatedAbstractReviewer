# AutomatedAbstractReviewer
Automation of classification of Medical Abstracts 

**Requirements**
1. Python
2. Scikit-learn

**Best Work Environment**
Install Canopy (for hassle free recursive installations of dependencies)

**Steps to Run**:

1. python preprocess.py input_xml_file output_file_name 
2. sh run_train_test_split.sh preprocessed_output train_output_file test_output_file
3. python generateDataWithLabels.py train/test_output_file Crowd_Workers_Label_fie(RawTurkResults.csv) ques_num(1,2,3,4) op_file
4. sh run_under_sampling.sh question_num
5. sh run_classification.sh question_num classifier_name op_file
6. python post_preprocess.py classify_op_q1 classify_op_q2 classify_op_q3 classify_op_4 gold_label_file(relevant.txt)

**Order of execution** is as follows:
* preprocess.py
* train_test_split.py
* generateDataWithLabels.py
* under_sampling.py
* classify.py
* post_process.py

I have included the shell scripts to run the python scripts. 

The **data** folder contains all the data (input and output) that is required for this project
* **preprocess_output**: This folder contains the output of preprocess.py 
* **train_test_data**: contains the output of train_test_split.py
* **data_max_voting**: output of generateDataWithLabels but output of when maximum voting procedure is performed to club answers from different crowd workers for the same abstract
* **data_repeat_rec**: output of generateDataWithLabels. This is the output when every record (no algorithm is used to join answers of the crowd workers) is repeated with its corresponding label
* **undersample_output**: output after undersampling is performed
* **classify_output**: output of the classifier. Has the predicted class variables for every abstract id in the test data
* **RawTurkResults.csv** : this is the input file which contains the different labels provided by the crowd workers for each question for every abstract. The key columns of interest are abstractid, question columns
* **proto-beam-all.xml**: this is the input file. the abstracts, the title, keywords and abstract id along with some other details about every abstract is presented in an xml format
* **relevant.txt**: Expert or gold labels. This file contains the abstract ids which were marked relavant by the experts. In other words, if an abstract id is not present in this file, then it is irrelevant to the subject of the expert
