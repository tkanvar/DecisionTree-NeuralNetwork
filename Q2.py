import numpy as np
import re
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys, os
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import math
import random
import time

def prepareLayersOfNN(noOfInputs, hiddenLayers, noOfOutputs):
	layers = []
	layers.append(noOfInputs)
	for i in range(len(hiddenLayers)):
		layers.append(hiddenLayers[i])
	layers.append(noOfOutputs)

	return layers

def initializeWB(layers):
	noOfLayers = len(layers)

	# w, b
	w = []
	b = []

	w.append(np.zeros((1,1)))		# 1st layer w doesnt exist
	b.append(np.zeros((1,1)))		# 1st layer b doesnt exist

	for i in range(1, noOfLayers):			# Input layer doesnt have w or b
		layerW = np.zeros((layers[i-1], layers[i]))

		noOfB = layers[i]
		layerB = np.zeros((noOfB,1))

		w.append(layerW)
		b.append(layerB)

	for i in range(1, len(w)):
		w[i] = np.random.uniform(size=(layers[i-1], layers[i]))

	for i in range(1, len(b)):
		b[i] = np.random.uniform(size=(layers[i], 1))

	return w, b

def initializeAZ(layers):
	noOfLayers = len(layers)

	# a, z
	a = []
	z = []

	z.append(np.zeros((1,1)))		# 1st layer z doesnt exist

	for i in range(noOfLayers):
		layerA = np.zeros((layers[i],1))
		a.append(layerA)

	for i in range(1, noOfLayers):
		layerZ = np.zeros((layers[i],1))
		z.append(layerZ)

	return a, z

def initializeDelta(layers):
	noOfLayers = len(layers)
	delta = []

	delta.append(np.zeros((1,1)))	# 1st layer delta doesnt exist

	for i in range(1, noOfLayers):
		layerDelta = np.zeros((layers[i],1))
		delta.append(layerDelta)

	return delta

def readData(trainfilename, testfilename):
	traindata = pd.read_table(trainfilename, delimiter=',', header=None)
	testdata = pd.read_table(testfilename, delimiter=',', header=None)

	trainX = traindata.iloc[:,:-1]
	testX = testdata.iloc[:,:-1]
	trainY = traindata.iloc[:,-1]
	testY = testdata.iloc[:,-1]

	return trainX, trainY, testX, testY

def shuffle(trainX, trainY):
	train = np.c_[trainX.reshape(len(trainX), -1), trainY.reshape(len(trainY), -1)]
	np.random.shuffle(train)
	trainX = train[:, :trainX.size//len(trainX)].reshape(trainX.shape)
	trainY = train[:, trainX.size//len(trainX):].reshape(trainY.shape)

	return trainX, trainY

def getOneHotEncodingX(trainX, testX):
	trainRows = trainX.shape[0]

	cols = []
	for i in range(trainX.shape[1]):
		cols.append(i)

	X = pd.concat([trainX, testX],axis=0)
	X = pd.get_dummies(X, columns=cols)

	trainX = X.iloc[:trainRows,:]
	testX = X.iloc[trainRows:,:]

	trainX1 = trainX.values
	testX1 = testX.values

	return trainX1, testX1

def getOneHotEncodingY(trainY, testY):
	trainRows = trainY.shape[0]

	X = pd.concat([trainY, testY],axis=0)
	X = pd.get_dummies(X, columns=[0])

	trainY = X.iloc[:trainRows,:]
	testY = X.iloc[trainRows:,:]

	trainY1 = trainY.values
	testY1 = testY.values

	return trainY1, testY1

def getBatch(min_batch_cntr, batchSize, trainX, trainY):
	startInd = min_batch_cntr * batchSize
	endInd = startInd + batchSize

	trainX1 = trainX[startInd:endInd, :]
	trainY1 = trainY[startInd:endInd, :]

	return trainX1, trainY1

def initializeBatchWB(layers):
	noOfLayers = len(layers)

	batchW = []
	batchB = []

	batchW.append(np.zeros((1,1)))		# 1st layer w doesnt exist
	batchB.append(np.zeros((1,1)))		# 1st layer b doesnt exist

	for i in range(1, noOfLayers):			# Output layer doesnt have w or b
		layerW = np.zeros((layers[i-1], layers[i]))
		layerB = np.zeros((layers[i],1))

		batchW.append(layerW)
		batchB.append(layerB)

	return batchW, batchB

def initializeInputLayer(a, trainRow):
	a[0] = trainRow
	return a

def sigmoid(z, useSigmoid, isLastLayer):
	a = 0.0
	if useSigmoid == True or isLastLayer == True:
		e = np.exp(-z)
		a = (1.0 / (1.0 + e))
	else:
		zerovec = np.zeros((z.shape[0], z.shape[1]))
		a = np.maximum(zerovec, z)
	return a

def forwardPropagate(w, b, z, a, layers, useSigmoid):
	for l in range(1, len(layers)):
		z[l] = (w[l].T @ a[l - 1]) + b[l]

		isLastLayer = False
		if l == (len(layers) - 1): 
			isLastLayer = True
		a[l] = sigmoid(z[l], useSigmoid, isLastLayer)

	return a, z

def derivativeSigmoid(a, useSigmoid, isLastLayer):
	r = 0.0
	if useSigmoid == True or isLastLayer == True:
		r = a * (1 - a)
	else:
		r = np.zeros((a.shape[0], a.shape[1]))
		np.place(r, a > 0, 1)
		np.place(r, a <= 0, 0)

	return r

def calculateOutputLayerDelta(delta, trainY, a, layers, useSigmoid):
	L = len(layers) - 1
	delta[L] = (a[L] - trainY) * derivativeSigmoid(a[L], useSigmoid, True)
	return delta

def backwardPropagate(w, b, a, delta, layers, useSigmoid):
	for l in range((len(layers) - 2), 0, -1):						# Going from second last layer to 2nd layer. Excluding 1st layer. For loop runs till l > 0
		delta[l] = (w[l+1] @ delta[l+1]) * derivativeSigmoid(a[l], useSigmoid, False)

	return delta

def updateDelWB(batchDelW, batchDelB, delta, a, layers):
	for l in range(1, len(layers)):
		batchDelW[l] = a[l - 1] @ delta[l].T
		batchDelB[l] = delta[l]

	return batchDelW, batchDelB

def updateC(trainY, a, layers):
	L = len(layers) - 1

	aL = a[L]

	batchC = np.sum(np.power((trainY - aL), 2)) / 2.0
	return batchC

def updateWB(w, batchDelW, b, batchDelB, learning_rate, layers, batchSize):
	for l in range(1, len(layers)):
		w[l] = w[l] - learning_rate * batchDelW[l]
		b[l] = b[l] - learning_rate * (np.sum(batchDelB[l], axis=1)).reshape((-1, 1))

	return w, b

def readAndOneHotEncodingData(trainfilename, testfilename, outtrainfilename = "", outtestfilename = ""):
	trainX, trainY, testX, testY = readData(trainfilename, testfilename)
	trainX, testX = getOneHotEncodingX(trainX, testX)
	trainY, testY = getOneHotEncodingY(trainY, testY)

	if outtrainfilename != "":
		train = np.concatenate((trainX, trainY), axis=1)
		np.savetxt(outtrainfilename, train, fmt='%d')

		test = np.concatenate((testX, testY), axis=1)
		np.savetxt(outtestfilename, test, fmt='%d')

	return trainX, trainY, testX, testY

def trainNN(trainX, trainY, maxepocs, batchSize, noOfInputs, hiddenLayers, noOfOutputs, threshold, learning_rate, tolerance, useSigmoid):
	# Variables
	layers = prepareLayersOfNN(noOfInputs, hiddenLayers, noOfOutputs)
	w, b = initializeWB(layers)
	a, z = initializeAZ(layers)
	delta = initializeDelta(layers)

	trainX, trainY = shuffle(trainX, trainY)
		
	# SGD
	isFirstLoop = True
	withinThresholdCounter = 0
	reachedOptimum = False
	pvsBatchC = 0.0
	curBatchC = 0.0
	pvsEpochC = 0.0
	curEpochC = 0.0
	isEpochLossFail = 0
	epoc = 0
	printStr = ""

	batchDelW, batchDelB = initializeBatchWB(layers)

	for epoc in range(maxepocs):
		withinThresholdCounter = 0

		for min_batch_cntr in range(int(len(trainX) / batchSize)):
			trainXBatch, trainYBatch = getBatch(min_batch_cntr, batchSize, trainX, trainY)

			a = initializeInputLayer(a, trainXBatch.T)
			a, z = forwardPropagate(w, b, z, a, layers, useSigmoid)
			delta = calculateOutputLayerDelta(delta, trainYBatch.T, a, layers, useSigmoid)
			delta = backwardPropagate(w, b, a, delta, layers, useSigmoid)

			batchDelW, batchDelB = updateDelWB(batchDelW, batchDelB, delta, a, layers)
			curBatchC = updateC(trainYBatch.T, a, layers)
			w, b = updateWB(w, batchDelW, b, batchDelB, learning_rate, layers, batchSize)

			if isFirstLoop == False and abs(pvsBatchC - curBatchC) <= threshold:
				withinThresholdCounter += 1
				if (withinThresholdCounter >= (int(len(trainX) / (batchSize * 2)))):
					reachedOptimum = True
					break
			else: isFirstLoop = False

			pvsBatchC = curBatchC

		curEpochC = curBatchC
		if tolerance != -1 and abs(pvsEpochC - curEpochC) <= tolerance:
			isEpochLossFail += 1
			if isEpochLossFail == 2:
				learning_rate = learning_rate / 5.0
				isEpochLossFail = 0;
		else:
			isEpochLossFail = 0
		pvsEpochC = curEpochC

		if reachedOptimum == True:
			break

	return w, b, layers

def testNN(x, y, w, b, layers, printStr, useSigmoid):
	a, z = initializeAZ(layers)
	actualY = []
	predY = []

	correct = 0
	total = 0
	for row in range(x.shape[0]):
		XRow = x[row]
		XRow = XRow.reshape((-1, 1))
		YRow = y[row]
		YRow = YRow.reshape((-1, 1))

		a = initializeInputLayer(a, XRow)
		a, z = forwardPropagate(w, b, z, a, layers, useSigmoid)

		L = len(layers) - 1
		aL = a[L]
		maxo = 0
		maxa = 0
		predOpIndex = -1
		actualIndex = -1
		for i in range(aL.shape[0]):
			if aL[i][0] > maxo:
				maxo = aL[i][0]
				predOpIndex = i
			if YRow[i][0] > maxa:
				maxa = YRow[i][0]
				actualIndex = i

		if predOpIndex == actualIndex: 
			correct += 1
		total += 1

		actualY.append(actualIndex)
		predY.append(predOpIndex)

	acc = (correct * 100 / total)
	print (printStr, " Accuracy = ",acc)

	return actualY, predY, acc

def confusionMatrix(actualY, predictedY, printStr):
	y_true = pd.Series(actualY, name='Actual')
	y_pred = pd.Series(predictedY, name='Predicted')
	
	print ("Confusion Matrix ", printStr)
	print (pd.crosstab(y_true, y_pred, margins=True))

def plot (timetaken, acctrain, acctest, hiddenlayer):
	plt.plot(hiddenlayer, timetaken, label="Time Taken to train NN", color="red")
	plt.plot(hiddenlayer, acctrain, label="Accuracy on Training Examples", color="green")
	plt.plot(hiddenlayer, acctest, label="Accuracy on Testing Examples", color="blue")

	plt.title("Single Hidden Layer NN")
	plt.xlabel("No of neurons in hidden layer")
	plt.ylabel("Data")

	plt.legend()
	plt.show()

def readConfig(configfile):
	f = open(configfile, "r")
	lines = f.readlines()
	f.close()

	noOfInputs = int(lines[0])
	noOfOutputs = int(lines[1])
	batchSize = int(lines[2])

	nhidden = int(lines[3])
	hiddenlayers = lines[4].split()
	hiddenLayersArr = [[]]
	for i in range(nhidden):
		hiddenLayersArr[0].append(int(hiddenlayers[i]))
	
	useSigmoid = True
	relsig   = re.sub(r"[\n\t\s]*", "", lines[5])
	if relsig == "relu":
		useSigmoid = False

	fixedLearningRate = False
	fixvar   = re.sub(r"[\n\t\s]*", "", lines[6])
	if fixvar == "fixed":
		fixedLearningRate = True

	return batchSize, noOfInputs, hiddenLayersArr, noOfOutputs, fixedLearningRate, useSigmoid

partnum = sys.argv[1] # prints var2
trainfilename = sys.argv[2] # prints python_script.py
testfilename = sys.argv[3] # prints var1
outtrainfilename = sys.argv[4]
outtestfilename = sys.argv[5]
configfile = sys.argv[6]

######## preprocess.sh
if partnum == "a":
	trainX, trainY, testX, testY = readAndOneHotEncodingData(trainfilename, testfilename, outtrainfilename, outtestfilename)
	print("Number of Features = ", trainX.shape[1])

######## nn.sh
elif partnum == "b":
	batchSize, noOfInputs, hiddenLayersArr, noOfOutputs, fixedLearningRate, useSigmoid = readConfig(configfile)

	maxepocs = 100
	threshold = 1e-14
	learning_rate = 0.1
	tolerance = -1
	if fixedLearningRate == False: 
		tolerance = 10e-6
	
	trainX, trainY, testX, testY = readAndOneHotEncodingData(trainfilename, testfilename, "", "")

	timetakenSingleLayer = []
	acctrainSingleLayer = []
	acctestSingleLayer = []
	hiddenlayerSingleLayer = []

	for i in range(len(hiddenLayersArr)):
		hiddenLayers = hiddenLayersArr[i]

		w, b, layers = trainNN(trainX, trainY, maxepocs, batchSize, noOfInputs, hiddenLayers, noOfOutputs, threshold, learning_rate, tolerance, useSigmoid)

		actualTrainY, predTrainY, trainAcc = testNN(trainX, trainY, w, b, layers, "Training Example", useSigmoid)
		actualTestY, predTestY, testAcc = testNN(testX, testY, w, b, layers, "Testing Example", useSigmoid)

		confusionMatrix(actualTrainY, predTrainY, "Training Example")
		confusionMatrix(actualTestY, predTestY, "Testing Example")

		timetakenSingleLayer.append(timee - times)
		acctrainSingleLayer.append(trainAcc)
		acctestSingleLayer.append(testAcc)
		hiddenlayerSingleLayer.append(hiddenLayers[0])
