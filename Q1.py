import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys, os
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def plot(nodes, train, test, valid, title, legend1, legend2, legend3, ylab, xlab, reversex):
	plt.plot(nodes, train, label=legend1, color="red")
	plt.plot(nodes, test, label=legend2, color="green")
	plt.plot(nodes, valid, label=legend3, color="blue")

	plt.title(title)
	plt.xlabel(xlab)
	plt.ylabel(ylab)

	if reversex == True:
		plt.xlim(max(nodes), 0)
	
	plt.legend()

	plt.show()

def populateDataFromOutfile(outfile):
	f = open(outfile, "r")
	lines = f.readlines()
	f.close()

	nodes = []
	train = []
	test = []
	valid = []
	sectionNo = 0

	for i in range(len(lines)):
		line = lines[i]
		if line == "\n":
			sectionNo += 1
			continue

		if sectionNo == 0:
			nodes.append(int(line))
		elif sectionNo == 1:
			train.append(float(line))
		elif sectionNo == 2:
			valid.append(float(line))
		elif sectionNo == 3:
			test.append(float(line))

	return nodes, train, test, valid

def populateData(trainfile, testfile, validfile):
	fileintrain = open(trainfile, "r")
	fileintest = open(testfile, "r")
	fileinvalid = open(validfile, "r")

	readertrain = csv.reader(fileintrain)
	readertest = csv.reader(fileintest)
	readervalid = csv.reader(fileinvalid)

	head = []
	trainX = []
	trainY = []
	testX = []
	testY = []
	validX = []
	validY = []

	lineno = 0
	for row in readertrain:
		lineno += 1
		if lineno == 1:	
			head.append(row)
			continue

		if lineno == 2:	continue

		tempX = []
		tempY = []
		for r in range(len(row)):
			if r == 0: continue
			elif r == (len(row) - 1): tempY.append(row[r])
			else: tempX.append(row[r])
		trainX.append(tempX)
		trainY.append(tempY)

	lineno = 0
	for row in readertest:
		lineno += 1
		if lineno == 1:	continue
		if lineno == 2:	continue

		tempX = []
		tempY = []
		for r in range(len(row)):
			if r == 0: continue
			elif r == (len(row) - 1): tempY.append(row[r])
			else: tempX.append(row[r])
		testX.append(tempX)
		testY.append(tempY)

	lineno = 0
	for row in readervalid:
		lineno += 1
		if lineno == 1:	continue
		if lineno == 2:	continue

		tempX = []
		tempY = []
		for r in range(len(row)):
			if r == 0: continue
			elif r == (len(row) - 1): tempY.append(row[r])
			else: tempX.append(row[r])
		validX.append(tempX)
		validY.append(tempY)

	return trainX, trainY, testX, testY, validX, validY

def readAndOneHotEncodeData(train, test, valid):
	traindata = pd.read_table(train, delimiter=',', header=None)
	testdata = pd.read_table(test, delimiter=',', header=None)
	validdata = pd.read_table(valid, delimiter=',', header=None)

	traindata = traindata.iloc[2:,1:]
	testdata = testdata.iloc[2:,1:]
	validdata = validdata.iloc[2:,1:]

	trainX = traindata.iloc[:,:-1]
	trainY = traindata.iloc[:,-1]

	testX = testdata.iloc[:,:-1]
	testY = testdata.iloc[:,-1]

	validX = validdata.iloc[:,:-1]
	validY = validdata.iloc[:,-1]

	trainRows = traindata.shape[0]
	testRows = trainRows + testdata.shape[0]

	cols = []
	for i in range(trainX.shape[1]):
		cols.append(i+1)

	X = pd.concat([trainX, testX, validX],axis=0)
	Xdata = pd.get_dummies(X, columns=cols)

	Y = pd.concat([trainY, testY, validY],axis=0)
	Ydata = pd.get_dummies(Y, columns=[0])
	
	trainX = Xdata[:trainRows]
	testX = Xdata[trainRows:testRows]	
	validX = Xdata[testRows:]	

	trainY = Ydata[:trainRows]
	testY = Ydata[trainRows:testRows]
	validY = Ydata[testRows:]

	return trainX, trainY, testX, testY, validX, validY

partnum = sys.argv[1] # prints var2
train = sys.argv[2] # prints python_script.py
test = sys.argv[3] # prints var1
valid = sys.argv[4] # prints var1

#train = "C:\\tanuk\\WorkArea\\IITD\\MS\\COL774\\Assign3\\DecisionTree\\data\\credit-cards.train.csv"
#test = "C:\\tanuk\\WorkArea\\IITD\\MS\\COL774\\Assign3\\DecisionTree\\data\\credit-cards.test.csv"
#valid = "C:\\tanuk\\WorkArea\\IITD\\MS\\COL774\\Assign3\\DecisionTree\\data\\credit-cards.val.csv"
#partnum = "e"

dir_path = os.path.dirname(os.path.realpath(__file__))
exename = dir_path+"/DecisionTree/DecisionTree.exe"

outfile = dir_path + "\\"

######## Part (a)
if partnum == "1":
	outfile += "part_a.txt"
	os.system(exename + " " + train + " " + valid + " " + test + " " + "a" + " " + outfile)

	nodes, train, test, valid = populateDataFromOutfile(outfile)
	plot(nodes, train, test, valid, "Constructing Decision Tree", "Training Data", "Testing Data", "Validation Data", "Accuracy", "Number of Data", False)

######## Part (b)
elif partnum == "2":
	outfile += "part_b.txt"
	os.system(exename + " " + train + " " + valid + " " + test + " " + "b" + " " + outfile)

	nodes, train, test, valid = populateDataFromOutfile(outfile)
	plot(nodes, train, test, valid, "Post Pruning Process", "Training Data", "Testing Data", "Validation Data", "Accuracy", "Number of Data", True)

######## Part (c)
elif partnum == "3":
	outfile += "part_c.txt"
	os.system(exename + " " + train + " " + valid + " " + test + " " + "c" + " " + outfile)

	nodes, train, test, valid = populateDataFromOutfile(outfile)
	plot(nodes, train, test, valid, "Constructing Tree without Median", "Training Data", "Testing Data", "Validation Data", "Accuracy", "Number of Data", False)

######## Part (d)
elif partnum == "4":
	trainX, trainY, testX, testY, validX, validY = populateData(train, test, valid)

	# Default Parameters
	decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0)
	decisionTree.fit(trainX, trainY)

	print("Training Accuracy: %.3f" % (100 * decisionTree.score(trainX, trainY)))
	print("Validation Accuracy: %.3f" % (100 * decisionTree.score(validX, validY)))
	print("Testing Accuracy: %.3f" % (100 * decisionTree.score(testX, testY)))

	# max_depth
	accuracy1 = []
	depths = range(2, 43)
	for d in depths:
		decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=d)
		decisionTree.fit(trainX, trainY)
		accuracy1.append(100 * decisionTree.score(validX, validY))

	plt.plot(depths, accuracy1)
	plt.xlabel("Depth of Tree")
	plt.ylabel("Validation accuracy1")
	plt.title("Validation Accuracy with depth of tree")
	plt.show()

	# Min Sample Split
	accuracy2 = []
	splits = range(2, 2500, 100)
	for d in splits:
		decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=d)
		decisionTree.fit(trainX, trainY)
		accuracy2.append(100 * decisionTree.score(validX, validY))

	plt.plot(splits, accuracy2)
	plt.xlabel("Minimum Samples to Split Node")
	plt.ylabel("Validation accuracy2")
	plt.title("Validation Accuracy with min_samples_split")
	plt.show()

	# Min Sample Leaves
	accuracy3 = []
	leaf = range(2, 2500, 100)
	for d in leaf:
		decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_leaf=d)
		decisionTree.fit(trainX, trainY)
		accuracy3.append(100 * decisionTree.score(validX, validY))

	plt.plot(leaf, accuracy3)
	plt.xlabel("Minimum Samples At Leaf Node")
	plt.ylabel("Validation accuracy3")
	plt.title("Validation Accuracy with min_samples_leaf")
	plt.show()

	# Best Validation Accuracy
	accuracy4 = []
	parameters = []
	leaf = range(2, 2500, 100)
	splits = range(2, 2500, 100)
	depths = range(2, 43)
	for d in leaf:
		for e in splits:
			for f in depths:
				decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_leaf=d,min_samples_split=e,max_depth=f)
				decisionTree.fit(trainX, trainY)

				temp = []
				temp.append(d)
				temp.append(e)
				temp.append(f)
				parameters.append(temp)
				accuracy4.append(100 * decisionTree.score(validX, validY))

	print (max(accuracy4))
	print (parameters[accuracy4.index(max(accuracy4))])

	# Best Accuracy
	decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0,min_samples_leaf=100,min_samples_split=802,max_depth=9)
	decisionTree.fit(trainX, trainY)

	print("Training Accuracy: %.3f" % (100 * decisionTree.score(trainX, trainY)))
	print("Validation Accuracy: %.3f" % (100 * decisionTree.score(validX, validY)))
	print("Testing Accuracy: %.3f" % (100 * decisionTree.score(testX, testY)))

######## Part (e)
elif partnum == "5":
	trainX, trainY, testX, testY, validX, validY = readAndOneHotEncodeData(train, test, valid)

	decisionTree = DecisionTreeClassifier(criterion="entropy", random_state=0,min_samples_leaf=100,min_samples_split=802,max_depth=9)
	decisionTree.fit(trainX, trainY)

	print("Training Accuracy: %.3f" % (100 * decisionTree.score(trainX, trainY)))
	print("Validation Accuracy: %.3f" % (100 * decisionTree.score(validX, validY)))
	print("Testing Accuracy: %.3f" % (100 * decisionTree.score(testX, testY)))

######## Part (f)
elif partnum == "6":
	trainX, trainY, testX, testY, validX, validY = readAndOneHotEncodeData(train, test, valid)

	# Default Parameters
	decisionTree = RandomForestClassifier(criterion="entropy", random_state=0)
	decisionTree.fit(trainX, trainY)

	print("Training Accuracy: %.3f" % (100 * decisionTree.score(trainX, trainY)))
	print("Validation Accuracy: %.3f" % (100 * decisionTree.score(validX, validY)))
	print("Testing Accuracy: %.3f" % (100 * decisionTree.score(testX, testY)))

	# estimators
	accuracy1 = []
	estimators = range(2, 100)
	for d in estimators:
		decisionTree = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=d)
		decisionTree.fit(trainX, trainY)
		accuracy1.append(100 * decisionTree.score(validX, validY))

	plt.plot(estimators, accuracy1)
	plt.xlabel("N_Estimators")
	plt.ylabel("Validation accuracy1")
	plt.title("Validation Accuracy with N_Estimators")
	plt.show()

	# Max features
	accuracy2 = []
	maxfeat = range(3, 90)
	for d in maxfeat:
		decisionTree = RandomForestClassifier(criterion="entropy", random_state=0, max_features=d)
		decisionTree.fit(trainX, trainY)
		accuracy2.append(100 * decisionTree.score(validX, validY))

	plt.plot(maxfeat, accuracy2)
	plt.xlabel("Maximum Features")
	plt.ylabel("Validation accuracy2")
	plt.title("Validation Accuracy with maximum features")
	plt.show()

	# bootstrap
	accuracy3 = []
	bootsrap = [True, False]
	for d in bootstrap:
		decisionTree = RandomForestClassifier(criterion="entropy", random_state=0, bootstrap=d)
		decisionTree.fit(trainX, trainY)
		accuracy3.append(100 * decisionTree.score(validX, validY))

	plt.plot(bootsrap, accuracy3)
	plt.xlabel("Bootstrap")
	plt.ylabel("Validation accuracy3")
	plt.title("Validation Accuracy with bootstrap")
	plt.show()

	# Best Validation Accuracy
	accuracy4 = []
	parameters = []
	estim = range(5, 25)
	maxfeat = range(3, 15)
	bootstrap = [True, False]
	for d in estim:
		for e in maxfeat:
			for f in bootstrap:
				decisionTree = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=d,max_features=e,bootstrap=f)
				decisionTree.fit(trainX, trainY)

				temp = []
				temp.append(d)
				temp.append(e)
				temp.append(f)
				parameters.append(temp)
				accuracy4.append(100 * decisionTree.score(validX, validY))

	print (max(accuracy4))
	print (parameters[accuracy4.index(max(accuracy4))])

	# Best Accuracy
	maxParam = parameters[accuracy4.index(max(accuracy4))]
	decisionTree = RandomForestClassifier(criterion="entropy", random_state=0,n_estimators=maxParam[0],max_features=maxParam[1],bootstrap=maxParam[2])
	decisionTree.fit(trainX, trainY)

	print("Training Accuracy: %.3f" % (100 * decisionTree.score(trainX, trainY)))
	print("Validation Accuracy: %.3f" % (100 * decisionTree.score(validX, validY)))
	print("Testing Accuracy: %.3f" % (100 * decisionTree.score(testX, testY)))
