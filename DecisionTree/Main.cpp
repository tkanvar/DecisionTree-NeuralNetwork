#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <map>
#include <thread>

using namespace std;

enum category
{
	boolean,
	categorical,
	output,
	continuos,
	none
};

class attribute
{
	string attributeName;
	string attributeId;
	int attributeIndex;
	int * median;				// If categorical attribute then median will be some value else NULL
	category ctgry;

	int * categoryValues;		// Xs attribute followed by Y attrubute is filled - i.e. X0 can take 3 values. So this field will be array of 3 values
	int categoryValuesSize;		// size telling size of categories

	attribute * next;

public:
	attribute()
	{
		attributeName = "";
		attributeId = "";
		median = NULL;
		ctgry = none;
		categoryValues = NULL;
		categoryValuesSize = 0;
		next = NULL;
		attributeIndex = -1;
	}

	int getNoOfCategories() { return categoryValuesSize; }
	string getAttributeName() { return attributeName; }
	string getAttributeId() { return attributeId; }
	int * getMedian() { return median; }
	category getCategory() { return ctgry; }
	int * getCategoryValues() { return categoryValues; }
	attribute * getNext() { return next; }
	int getAttributeIndex() { return attributeIndex; }

	void setAttributeIndex(int ind) { attributeIndex = ind; }
	void setAttributeName(string attrn) { attributeName = attrn; }
	void setAttributeId(string attri) { attributeId = attri; }
	void setMedian(int * med) { median = med; }
	void setCategory(category c) { ctgry = c; }
	void setSizeOfCategoryValues(int sz) 
	{ 
		categoryValuesSize = sz; 
		categoryValues = (int *)malloc(sizeof(int) * sz);
	}
	void setCategoryValue(int val, int cntr) { categoryValues[cntr] = val; }
	void setNext(attribute * n) { next = n; }
};

class dataRow
{
	int * attrValues;			// X followed by Y attribute
	int noOfAttributes;

	dataRow * next;

public:
	dataRow()
	{
		attrValues = NULL;
		noOfAttributes = 0;
		next = NULL;
	}

	int getOutputValue() { return attrValues[noOfAttributes-1]; }
	int * getValues() { return attrValues; }
	int getNoOfAttributes() { return noOfAttributes; }
	dataRow * getNext() { return next; }

	void setNext(dataRow * n) { next = n; }
	void setValue(int val, int cntr) { attrValues[cntr] = val; }
	void setNoOfAttributes(int sz)
	{
		noOfAttributes = sz;
		attrValues = (int *)malloc(sizeof(int) * sz);
	}
};

enum logicalCondition
{
	greaterThan,
	lessThanEqualTo,
	equalEqualTo,
	noneCond
};

class treeNode
{
	attribute * attrNode;			// Attribute of this node in the tree
	int attributeIndex;				// AttributeIndex of this node in the tree
	
	int * condition;				// Conditions on all the branch of this node
	int noOfBranches;		// No of branches of the child
	treeNode ** children;			// Array of children nodes based on the conditions of the node
	treeNode * parent;

	dataRow ** trainDataArrAtNode;		// Array to pointer to training data rows that fall under this node
	int noOfTrainData;				// Number of training data at this node
	int decisionTrueCount;			// Number of True conclusions at this node
	int decisionFalseCount;			// Number of False conclusions at this node

	logicalCondition * logCond;

public:
	~treeNode()
	{
		logCond = NULL;
		logCond = 0;
		attributeIndex = -1;
		attrNode = NULL;
		attrNode = 0;
		condition = 0;
		condition = NULL;
		noOfBranches = 0;
		delete children;
		children = NULL;
		children = 0;
		parent = NULL;
		parent = 0;
		delete trainDataArrAtNode;
		trainDataArrAtNode = NULL;
		trainDataArrAtNode = 0;
		noOfTrainData = 0;
		decisionFalseCount = 0;
		decisionTrueCount = 0;
	}
	treeNode()
	{
		parent = NULL;
		attrNode = NULL;
		attributeIndex = -1;
		children = NULL;
		trainDataArrAtNode = NULL;
		decisionFalseCount = 0;
		decisionTrueCount = 0;
		condition = NULL;
		noOfBranches = 0;
		logCond = NULL;
	}

	void deepCopy(treeNode * newNode)
	{
		delete newNode->children;
		newNode->children = 0;
		newNode->children = NULL;
		delete newNode->trainDataArrAtNode;
		newNode->trainDataArrAtNode = NULL;
		newNode->trainDataArrAtNode = 0;

		newNode->logCond = logCond;
		newNode->setParentNode(parent);
		newNode->setAttrNode(attrNode);
		newNode->setAttributeIndex(attributeIndex);
		newNode->setNoOfBranches(noOfBranches);
		newNode->setCondition(condition);
		newNode->setDecisionFalseCount(decisionFalseCount);
		newNode->setDecisionTrueCount(decisionTrueCount);
		newNode->setNoOfTrainData(noOfTrainData);

		for (int i = 0; i < noOfBranches; i++)
			newNode->setChild(children[i], i);
		for (int i = 0; i < noOfTrainData; i++)
			newNode->setTrainDataRowAtNode(trainDataArrAtNode[i], i);
	}

	void setLogicalCondition(logicalCondition * lc) { logCond = lc; }
	void setBranchesSize(int b) { noOfBranches = b; }
	void setCondition(int * c) { condition = c; }
	void setDecisionFalseCount(int f) { decisionFalseCount = f; }
	void setDecisionTrueCount(int t) { decisionTrueCount = t; }
	void setTrainDataRowsAtNode(dataRow ** tdan) { trainDataArrAtNode = tdan; }
	void setChildren(treeNode ** cldrn) { children = cldrn; }
	dataRow ** getTrainDataArrAtNode() { return trainDataArrAtNode; }
	int getNoOfBranches() { return noOfBranches; }
	logicalCondition getLogicalCondition(int ind) { return logCond[ind]; }
	int getCondition(int ind) { return condition[ind]; }
	attribute * getAttrNode() { return attrNode; }
	int getAttributeIndex() { return attributeIndex; }
	treeNode * getChild(int ind) { return children[ind]; }
	dataRow * getTrainDataRowAtNode(int ind) { return trainDataArrAtNode[ind]; }
	int getNoOfTrainData() { return noOfTrainData; }
	int getNoOfDecisionTrueAtNode() { return decisionTrueCount; }
	int getNoOfDecisionFalseAtNode() { return decisionFalseCount; }
	treeNode * getParentNode() { return parent; }

	void setParentNode(treeNode * t) { parent = t; }
	void setNoOfBranches(int n)
	{
		noOfBranches = n;
		condition = (int *)malloc(sizeof(int) * n);
		logCond = (logicalCondition *)malloc(sizeof(logicalCondition) * n);
		setNoOfChildren(n);
	}
	void setBranchCondition(int cond, int ind) { condition[ind] = cond; }
	void setBranchLogicalCon(logicalCondition lc, int ind) { logCond[ind] = lc; }

	void storeTrainData(dataRow * trainDataHead, int noOfTrainingData)
	{
		setNoOfTrainData(noOfTrainingData);

		int cntr = 0;
		for (dataRow * trainDataTemp = trainDataHead; trainDataTemp->getNext() != NULL; trainDataTemp = trainDataTemp->getNext(), cntr++)
			setTrainDataRowAtNode(trainDataTemp, cntr);
	}
	void storeTrainData(dataRow ** trainDataRowArr, int noOfRows)
	{
		setNoOfTrainData(noOfRows);
		for (int i = 0; i < noOfRows; i++)
			setTrainDataRowAtNode(trainDataRowArr[i], i);
	}
	void setTrainDataSize(int n) { noOfTrainData = n; }
	void setNoOfTrainData(int n)
	{
		noOfTrainData = n;
		trainDataArrAtNode = (dataRow **)malloc(sizeof(dataRow *) * n);
	}
	void setTrainDataRowAtNode(dataRow * trainDataRow, int cntr) { trainDataArrAtNode[cntr] = trainDataRow; }
	void setNoOfTrueOutputs(int cnt) { decisionTrueCount = cnt; }
	void setNoOfFalseOutputs(int cnt) { decisionFalseCount = cnt; }
	void setAttrNode(attribute * attr) { attrNode = attr; }
	void setAttributeIndex(int attrInd) { attributeIndex = attrInd; }
	void countNoOfTrueFalseDecisions()
	{
		for (int i = 0; i < noOfTrainData; i++)
		{
			if (trainDataArrAtNode[i]->getOutputValue() == 0) decisionFalseCount++;
			else decisionTrueCount++;
		}
	}
	void setNoOfChildren(int n)
	{
		noOfBranches = n;
		children = (treeNode **)malloc(sizeof(treeNode *) * n);
	}
	void setChild(treeNode * child, int cntr) { children[cntr] = child; }

	void print(int level)
	{
		cout << "\n";
		string strTabs = "";
		for (int i = 0; i < level; i++) strTabs += "\t";

		cout << strTabs << "Level = " << level << ":";
		if (attrNode != NULL)
			cout << attrNode->getAttributeName() << ":" << attrNode->getAttributeId() << ":";
		cout << "No Of Children = " << noOfBranches << ":";
		cout << "# of False = " << decisionFalseCount << ":";
		cout << "# of True = " << decisionTrueCount << ":";

		for (int i = 0; i < noOfBranches; i++)
		{
			treeNode * child = children[i];
			child->print(level+1);
		}
	}
};

struct AccuracyData
{
	int nodeNumber;
	float trainAccuracy;
	float validAccuracy;
	float testAccuracy;

	AccuracyData * next;

	AccuracyData()
	{
		nodeNumber = -1;
		trainAccuracy = 0;
		validAccuracy = 0;
		testAccuracy = 0;
		next = NULL;
	}
};

class Accuracy
{
	AccuracyData * accdatahead;			// This is head of AccuracyData
	AccuracyData * accdatatail;			// This is the cur accuracyData

	dataRow * validDataHead;
	int noOfValidData;
	dataRow * testDataHead;
	int noOfTestData;
	dataRow * trainDataHead;
	int noOfTrainData;

	string partnum;						// Q1 partnumber in assignment

public:
	Accuracy()
	{
		accdatahead = new AccuracyData();
		accdatatail = accdatahead;
		testDataHead = NULL;
		trainDataHead = NULL;
		validDataHead = NULL;
		noOfTestData = 0;
		noOfTrainData = 0;
		noOfValidData = 0;
	}

	AccuracyData * getAccuracyData() { return accdatahead; }

private:
	int * getPredictedOutput(treeNode * treeNodeHead, dataRow * dataHead, int noOfDataRows, string typeOfData, int & noOfDataRowsConsidered)			// noOfDataRowsConsidered might be less if there are rows where data is present that was not in train data and hence decision tree doesnt consider those cases
	{
		//cout << "\nTree\n";
		//treeNodeHead->print(1);

		treeNode * currTreeNode = NULL, * chosenChildTreeNode = NULL;
		noOfDataRowsConsidered = 0;

		int * predictedOutputCorrect = (int *)malloc(sizeof(int) * noOfDataRows);

		int cntr = 0;
		for (dataRow * dataRow = dataHead; dataRow->getNext() != NULL; dataRow = dataRow->getNext(), cntr++)
		{
			for (currTreeNode = treeNodeHead; currTreeNode != NULL;)
			{
				int indx = currTreeNode->getAttributeIndex();
				int * values = dataRow->getValues();
				int valCorespToAttr = values[indx];				// Get the value corresponding to the attribute A in the dataRow. attribute A is treeNodeTemp

				chosenChildTreeNode = currTreeNode;
				int i;
				for (i = 0; i < currTreeNode->getNoOfBranches(); i++)
				{
					if ((currTreeNode->getLogicalCondition(i) == equalEqualTo && currTreeNode->getCondition(i) == valCorespToAttr) ||
						(currTreeNode->getLogicalCondition(i) == greaterThan && currTreeNode->getCondition(i) < valCorespToAttr) ||
						(currTreeNode->getLogicalCondition(i) == lessThanEqualTo && currTreeNode->getCondition(i) >= valCorespToAttr))
					{
						chosenChildTreeNode = currTreeNode->getChild(i);
						break;
					}
				}

				if (i != 0 && i == currTreeNode->getNoOfBranches())
				{
					predictedOutputCorrect[cntr] = 0;
					//cout << "Data set wrong. Attribute value is not recorded in the treeNode: " << valCorespToAttr << " for attribute: " << currTreeNode->getAttrNode()->getAttributeName() << ". This is in " << typeOfData << " data.\n";
					break;
				}

				if (chosenChildTreeNode->getNoOfBranches() == 0)
				{
					noOfDataRowsConsidered++;

					if (chosenChildTreeNode->getNoOfDecisionFalseAtNode() > chosenChildTreeNode->getNoOfDecisionTrueAtNode())
						predictedOutputCorrect[cntr] = 0;
					else
						predictedOutputCorrect[cntr] = 1;

					if (predictedOutputCorrect[cntr] == dataRow->getOutputValue()) predictedOutputCorrect[cntr] = 1;
					else predictedOutputCorrect[cntr] = 0;

					break;
				}

				currTreeNode = chosenChildTreeNode;
			}
		}

		return predictedOutputCorrect;
	}
	float getAccuracy(treeNode * headTreeNode, dataRow * dataHead, int noOfDataRows, string typeOfData)
	{
		int noOfDataRowsConsidered = 0;
		int * predictedOutputCorrect = getPredictedOutput(headTreeNode, dataHead, noOfDataRows, typeOfData, noOfDataRowsConsidered);
		int i = 0;
		int noOfCorrect = 0;
		for (int i = 0; i < noOfDataRows; i++)
		{
			if (predictedOutputCorrect[i] == 1) noOfCorrect++;
		}
		float accuracy = (float)noOfCorrect * 100 / (float)noOfDataRowsConsidered;

		delete predictedOutputCorrect;
		predictedOutputCorrect = 0;
		predictedOutputCorrect = NULL;

		return accuracy;
	}
	float calculateAccuracy(int nodeNo, treeNode * headTreeNode, dataRow * dataHead, int noOfDataRows, string typeOfData)
	{
		if (accdatatail->nodeNumber == -1)
			accdatatail->next = new AccuracyData();
		else if (accdatatail->nodeNumber != nodeNo)
		{
			accdatatail->next->next = new AccuracyData();
			accdatatail = accdatatail->next;
		}

		accdatatail->nodeNumber = nodeNo;

		return getAccuracy(headTreeNode, dataHead, noOfDataRows, typeOfData);
	}

public:
	string getPartNum() { return partnum; }
	int getNoOfValidData() { return noOfValidData; }
	int getNoOfTestData() { return noOfTestData; }
	int getNoOfTrainData() { return noOfTrainData; }

	void setPartNum(string p) { partnum = p; }
	void setNoOfTrainData(int n) { noOfTrainData = n; }
	void setNoOfTestData(int n) { noOfTestData = n; }
	void setNoOfValidData(int n) { noOfValidData = n; }

	void setValidData(dataRow * validDH) { validDataHead = validDH; }
	void setTrainData(dataRow * tdh) { trainDataHead = tdh; }
	void setTestData(dataRow * tdh) { testDataHead = tdh; }

	void calculateTrainAccuracy(int nodeNo, treeNode * headTreeNode)
	{
		accdatatail->trainAccuracy = calculateAccuracy(nodeNo, headTreeNode, trainDataHead, noOfTrainData, "train");
	}
	void calculateTestAccuracy(int nodeNo, treeNode * headTreeNode)
	{
		accdatatail->testAccuracy = calculateAccuracy(nodeNo, headTreeNode, testDataHead, noOfTestData, "test");
	}
	void calculateValidAccuracy(int nodeNo, treeNode * headTreeNode)
	{
		accdatatail->validAccuracy = calculateAccuracy(nodeNo, headTreeNode, validDataHead, noOfValidData, "valid");
	}
	float calculateAccuracy(treeNode * headTreeNode, dataRow * data, int noOfDataRows)
	{
		return getAccuracy(headTreeNode, data, noOfDataRows, "");
	}
};

///////////////// FUNCTIONS

int split(string line, char spliton, string splitdata[])
{
	int i = 0;
	while (line != "")
	{
		int splitonind = line.find(spliton);
		string word = line.substr(0, (splitonind == -1 ? line.size() : splitonind));
		line = (splitonind == -1 ? "" : line.substr(splitonind + 1, line.size()));

		splitdata[i++] = word;
	}

	return i;
}

int split(string line, char spliton, int splitdata[])
{
	int i = 0;
	while (line != "")
	{
		int splitonind = line.find(spliton);
		string word = line.substr(0, (splitonind == -1 ? line.size() : splitonind));
		line = (splitonind == -1 ? "" : line.substr(splitonind + 1, line.size()));

		splitdata[i++] = atoi(word.c_str());
	}

	return i;
}

int fillData(dataRow * dataLL, char * filename, attribute * attrsLL = NULL)
{
	std::ifstream ifbuf;
	ifbuf.open(filename, ios::in);

	string attrIdLine, attrNamesLine;
	getline(ifbuf, attrIdLine);
	getline(ifbuf, attrNamesLine);
	if (attrsLL != NULL)
	{
		string splitattrid[25], splitattrname[25];
		int noOfAttrs = split(attrIdLine, ',', splitattrid);
		split(attrNamesLine, ',', splitattrname);

		attribute * attrTemp = attrsLL;
		for (int i = 1; i < noOfAttrs; i++)				// start from 1 to ignore X0 attribute
		{
			attrTemp->setAttributeName(splitattrname[i]);
			attrTemp->setAttributeId(splitattrid[i]);
			attrTemp->setAttributeIndex(i - 1);
			
			if (splitattrid[i] == "X1" || splitattrid[i] == "X5" || splitattrid[i] == "X12" || splitattrid[i] == "X13" || splitattrid[i] == "X14" || splitattrid[i] == "X15" || splitattrid[i] == "X16" || splitattrid[i] == "X17" || splitattrid[i] == "X18" || splitattrid[i] == "X19" || splitattrid[i] == "X20" || splitattrid[i] == "X21" || splitattrid[i] == "X22" || splitattrid[i] == "X23")
				attrTemp->setCategory(continuos);
			else if (splitattrid[i] == "X2")
				attrTemp->setCategory(boolean);
			else if (splitattrid[i] == "X3" || splitattrid[i] == "X4" || splitattrid[i] == "X6" || splitattrid[i] == "X7" || splitattrid[i] == "X8" || splitattrid[i] == "X9" || splitattrid[i] == "X10" || splitattrid[i] == "X11")
				attrTemp->setCategory(categorical);

			attrTemp->setNext(new attribute());
			attrTemp = attrTemp->getNext();
		}
	}

	int cnt = 0;
	dataRow * tempdata = dataLL;
	string line;
	while (getline(ifbuf, line))
	{
		int splitData[25];
		int sz = split(line, ',', splitData);

		tempdata->setNoOfAttributes(sz-1);					// -1 is to ignore X0 attribute
		for (int i = 1; i < sz; i++)						// start from 1 to ignore X0 attribute
			tempdata->setValue(splitData[i], i-1);

		tempdata->setNext(new dataRow());
		tempdata = tempdata->getNext();
		cnt++;
	}

	ifbuf.close();
	return cnt;
}

// Preprocess

int qsortcompare(const void* a, const void* b)
{
	const int* x = (int*)a;
	const int* y = (int*)b;

	if (*x > *y)
		return 1;
	else if (*x < *y)
		return -1;

	return 0;
}

void calculateMedian(dataRow ** dataArr, int * median, int attrIndWorkOn, int noOfDataRows)
{
	// create and Sort array 
	int n = noOfDataRows;

	int * a = (int *)malloc(sizeof(int) * n);
	for (int i = 0; i < noOfDataRows; i++)
		a[i] = dataArr[i]->getValues()[attrIndWorkOn];
	qsort(a, n, sizeof(int), qsortcompare);

	// check for even case 
	if (n % 2 != 0)
		(*median) = a[n / 2];
	else
		(*median) = (a[(n - 1) / 2] + a[n / 2]) / 2.0;

	delete a;
	a = NULL;
	a = 0;
}

void calculateMedian(dataRow * dataHead, int * median, int attrIndWorkOn, int noOfDataRows)
{
	// create and Sort array 
	int n = noOfDataRows;

	int * a = (int *)malloc(sizeof(int) * n);
	int i = 0;
	for (dataRow * ltempdatarow = dataHead; ltempdatarow->getNext() != NULL; ltempdatarow = ltempdatarow->getNext())
		a[i++] = ltempdatarow->getValues()[attrIndWorkOn];
	qsort(a, n, sizeof(int), qsortcompare);

	// check for even case 
	if (n % 2 != 0)
		(*median) = a[n / 2];
	else
		(*median) = (a[(n - 1) / 2] + a[n / 2]) / 2.0;

	delete a;
	a = NULL;
	a = 0;
}

void adjustDataAccordingToMedian(dataRow * dataHead, int * median, int attrIndWorkOn)
{
	for (dataRow * ltempdata = dataHead; ltempdata->getNext() != NULL; ltempdata = ltempdata->getNext())
	{
		if (ltempdata->getValues()[attrIndWorkOn] > (*median))
			ltempdata->setValue(1, attrIndWorkOn);
		else
			ltempdata->setValue(0, attrIndWorkOn);
	}
}

void preprocessData(dataRow * dataHead, attribute * attrsHead, int noOfDataRows, bool processHead, char * partnum)
{
	attribute * ltempattr = attrsHead;
	for (int i = 0; i < dataHead->getNoOfAttributes(); i++)
	{
		if (_stricmp(partnum, "c") != 0 && (ltempattr->getMedian() != NULL || ltempattr->getCategory() == continuos))		// if none/boolean/categorical then no need for preprocess
		{
			dataRow * ltempdatarow = dataHead;
			int * median = ltempattr->getMedian();

			if (median == NULL)			// This means data is training data
			{
				median = new int();
				calculateMedian(ltempdatarow, median, i, noOfDataRows);
				ltempattr->setMedian(median);
			}

			adjustDataAccordingToMedian(ltempdatarow, median, i);
			if (processHead) ltempattr->setCategory(boolean);
		}
		
		if (processHead && (ltempattr->getCategory() == boolean || ltempattr->getCategory() == categorical))
		{
			map<int, int> * ltempmap = new map<int, int>;
			int noOfCategories = 0;
			for (dataRow * ltempdatarow = dataHead; ltempdatarow->getNext() != NULL; ltempdatarow = ltempdatarow->getNext())
			{
				if ((*ltempmap)[ltempdatarow->getValues()[i]] != 1)
				{
					(*ltempmap)[ltempdatarow->getValues()[i]] = 1;
					noOfCategories++;
				}
			}

			ltempattr->setSizeOfCategoryValues(noOfCategories);
			int j = 0;
			for (map<int, int>::iterator it = ltempmap->begin(); it != ltempmap->end(); it++)
			{
				if (it->second == 1)
					ltempattr->setCategoryValue(it->first, j++);
			}
		}

		ltempattr = ltempattr->getNext();
	}
}

// Decision Tree
int countNumberOfTypeOutput(dataRow ** trainDataArr, int noOfTrainData, int type)
{
	int cnt = 0;
	for (int i = 0; i < noOfTrainData; i++)
	{
		if (trainDataArr[i]->getOutputValue() == type)
			cnt++;
	}

	return cnt;
}

double calculateEntropy(double p, double n)
{
	if (p == 0 && n == 0)
		return 0;
	else if (p == 0)
		return ((-n / (p + n)) * log(n / (p + n)) / log(2));
	else if (n == 0)
		return ((-p / (p + n)) * log(p / (p + n)) / log(2));
	else
		return ((-p / (p + n)) * log(p / (p + n)) / log(2)) + ((-n / (p + n)) * log(n / (p + n)) / log(2));
}

double calculateInformationGain(int attrIndexToWorkOn, attribute * attrToWorkOn, dataRow ** trainDataArr, int noOfTrainData)
{
	double p = (double)(countNumberOfTypeOutput(trainDataArr, noOfTrainData, 1));
	double n = (double)(countNumberOfTypeOutput(trainDataArr, noOfTrainData, 0));

	double entropy = calculateEntropy(p, n);
	double expectedEntropy = 0.0;

	if (attrToWorkOn->getCategory() == continuos)
	{
		int median = 0;
		calculateMedian(trainDataArr, &median, attrToWorkOn->getAttributeIndex(), noOfTrainData);

		for (int i = 0; i < 2; i++)
		{
			int p1 = 0, n1 = 0;
			for (int j = 0; j < noOfTrainData; j++)
			{
				if ((i == 0 && trainDataArr[j]->getValues()[attrIndexToWorkOn] > median) || 
					(i == 1 && trainDataArr[j]->getValues()[attrIndexToWorkOn] <= median))
				{
					if (trainDataArr[j]->getOutputValue() == 0) n1++;
					else p1++;
				}
			}

			expectedEntropy += (((double(p1) + double(n1)) / (p + n)) * calculateEntropy(double(p1), double(n1)));
		}
	}
	else
	{
		for (int i = 0; i < attrToWorkOn->getNoOfCategories(); i++)
		{
			int condition = attrToWorkOn->getCategoryValues()[i];
			int p1 = 0, n1 = 0;

			for (int j = 0; j < noOfTrainData; j++)
			{
				if (trainDataArr[j]->getValues()[attrIndexToWorkOn] == condition)
				{
					if (trainDataArr[j]->getOutputValue() == 0) n1++;
					else p1++;
				}
			}

			expectedEntropy += (((double(p1) + double(n1)) / (p + n)) * calculateEntropy(double(p1), double(n1)));
		}
	}

	double infoGain = entropy - expectedEntropy;
	return infoGain;
}

bool canChooseAttribute(attribute * attr, treeNode * curTreeNode)
{
	if (attr->getCategory() == continuos)
		return true;

	for (; curTreeNode != NULL && curTreeNode->getAttrNode() != NULL; curTreeNode = curTreeNode->getParentNode())
	{
		if (attr->getAttributeId() == curTreeNode->getAttrNode()->getAttributeId())
			return false;
	}
	return true;
}

attribute * chooseBestAttribute(treeNode * curTreeNode, int & attrIndexChosen, attribute *** attrsArr, int & noOfAttrs, dataRow ** trainDataArr, int noOfTrainData)
{
	attribute * chosenAttr = NULL;
	attrIndexChosen = 0;
	double infoGain = 0.0;

	for (int i = 0; i < noOfAttrs; i++)
	{
		double temp = calculateInformationGain(i, (*attrsArr)[i], trainDataArr, noOfTrainData);
		if (infoGain < temp)
		{
			if (!canChooseAttribute((*attrsArr)[i], curTreeNode))
				continue;

			infoGain = temp;
			chosenAttr = (*attrsArr)[i];
			attrIndexChosen = chosenAttr->getAttributeIndex();
		}
	}

	return chosenAttr;
}

dataRow ** getSelectedDataRowArr(int & noOfDataRowArrEle, logicalCondition lc, int condition, int attributeIndex, dataRow ** trainDataRowsArr, int noOfTrainDataRows)
{
	noOfDataRowArrEle = 0;
	for (int i = 0; i < noOfTrainDataRows; i++)
	{
		if ((lc == equalEqualTo && trainDataRowsArr[i]->getValues()[attributeIndex] == condition) || 
			(lc == greaterThan && trainDataRowsArr[i]->getValues()[attributeIndex] > condition) ||
			(lc == lessThanEqualTo && trainDataRowsArr[i]->getValues()[attributeIndex] <= condition))
			noOfDataRowArrEle++;
	}
	
	dataRow ** selectedDataRowArr = (dataRow **)malloc(sizeof(dataRow *) * noOfDataRowArrEle);
	for (int i = 0, j = 0; i < noOfTrainDataRows; i++)
		if ((lc == equalEqualTo && trainDataRowsArr[i]->getValues()[attributeIndex] == condition) ||
			(lc == greaterThan && trainDataRowsArr[i]->getValues()[attributeIndex] > condition) ||
			(lc == lessThanEqualTo && trainDataRowsArr[i]->getValues()[attributeIndex] <= condition))
		{
			selectedDataRowArr[j] = trainDataRowsArr[i];
			j++;
		}

	return selectedDataRowArr;
}

bool constructDecisionTree(attribute ** attrsArr, int & noOfAttrs, treeNode * decisionTreeNode, treeNode * curDecisionTreeNode, Accuracy * accuracy, int & constructingNodeNo)
{
	if (curDecisionTreeNode->getNoOfDecisionFalseAtNode() == curDecisionTreeNode->getNoOfTrainData())			// Means all output values are either true or false
		return false;
	if (curDecisionTreeNode->getNoOfDecisionTrueAtNode() == curDecisionTreeNode->getNoOfTrainData())
		return false;

	int attrIndexChosen = -1;
	attribute * chosenAttr = chooseBestAttribute(curDecisionTreeNode, attrIndexChosen, &attrsArr, noOfAttrs, curDecisionTreeNode->getTrainDataArrAtNode(), curDecisionTreeNode->getNoOfTrainData());

	if (chosenAttr == NULL)
		return false;

	//if (constructingNodeNo % 100 == 0)
		//cerr << "Constructing node " << constructingNodeNo << "\n";

	curDecisionTreeNode->setAttrNode(chosenAttr);
	curDecisionTreeNode->setAttributeIndex(attrIndexChosen);

	int median = 0;
	int noOfChildren;
	if (chosenAttr->getCategory() == continuos)
	{
		noOfChildren = 2;
		calculateMedian(curDecisionTreeNode->getTrainDataArrAtNode(), &median, chosenAttr->getAttributeIndex(), curDecisionTreeNode->getNoOfTrainData());
	}
	else
		noOfChildren = chosenAttr->getNoOfCategories();

	curDecisionTreeNode->setNoOfBranches(noOfChildren);

	for (int i = 0; i < noOfChildren; i++)
	{
		// Set branch conditions
		if (chosenAttr->getCategory() == continuos)
		{
			if (i == 0) curDecisionTreeNode->setBranchLogicalCon(greaterThan, i);
			else curDecisionTreeNode->setBranchLogicalCon(lessThanEqualTo, i);

			curDecisionTreeNode->setBranchCondition(median, i);
		}
		else
		{
			curDecisionTreeNode->setBranchLogicalCon(equalEqualTo, i);
			curDecisionTreeNode->setBranchCondition(chosenAttr->getCategoryValues()[i], i);
		}

		// Set Other data
		int noOfDataRowArrEle = 0;
		dataRow ** selectedDataRowArr = getSelectedDataRowArr(noOfDataRowArrEle, curDecisionTreeNode->getLogicalCondition(i), curDecisionTreeNode->getCondition(i), attrIndexChosen, curDecisionTreeNode->getTrainDataArrAtNode(), curDecisionTreeNode->getNoOfTrainData());

		treeNode * child = new treeNode();
		child->storeTrainData(selectedDataRowArr, noOfDataRowArrEle);
		child->countNoOfTrueFalseDecisions();
		curDecisionTreeNode->setChild(child, i);

		delete selectedDataRowArr;
		selectedDataRowArr = 0;
		selectedDataRowArr = NULL;
	}

	if (accuracy->getPartNum() == "a" || accuracy->getPartNum() == "c")
	{
		accuracy->calculateTrainAccuracy(constructingNodeNo, decisionTreeNode);
		accuracy->calculateTestAccuracy(constructingNodeNo, decisionTreeNode);
		accuracy->calculateValidAccuracy(constructingNodeNo, decisionTreeNode);
	}

	for (int i = 0; i < noOfChildren; i++)
	{
		constructingNodeNo++;

		curDecisionTreeNode->getChild(i)->setParentNode(curDecisionTreeNode);
		bool lBIsAttributeChosen = constructDecisionTree(attrsArr, noOfAttrs, decisionTreeNode, curDecisionTreeNode->getChild(i), accuracy, constructingNodeNo);
		
		if (!lBIsAttributeChosen)
			constructingNodeNo--;
	}

	return true;
}


///////////////// Post Pruning 
void getNumberOfNodes(treeNode * curTreeNode, int & count)
{
	if (curTreeNode->getNoOfBranches() == 0) return;
	count++;		// Dont count leaf nodes

	for (int i = 0; i < curTreeNode->getNoOfBranches(); i++)
		getNumberOfNodes(curTreeNode->getChild(i), count);
}

struct bestPruneNode
{
	treeNode * nodeToPrune;
	float validAcc;
};

void getBestTreeNodeToPrune(treeNode * decisionTreeHead, treeNode * decisionTreeNode, Accuracy * accuracy, dataRow * validDataHead, int noOfValidData, int noOfNodesInTree, treeNode * tempCurNode, bestPruneNode * bestPruneNodeStruct)
{
	for (int i = 0; i < decisionTreeNode->getNoOfBranches(); i++)
	{
		treeNode * childNode = decisionTreeNode->getChild(i);
		if (childNode->getNoOfBranches() == 0)		// Leave leaf child nodes that have no attribute assigned to it
			continue;

		getBestTreeNodeToPrune(decisionTreeHead, childNode, accuracy, validDataHead, noOfValidData, noOfNodesInTree, tempCurNode, bestPruneNodeStruct);

		// Pruning
		childNode->deepCopy(tempCurNode);

		childNode->setAttrNode(NULL);
		childNode->setAttributeIndex(-1);
		childNode->setChildren(NULL);
		childNode->setLogicalCondition(NULL);
		childNode->setCondition(NULL);
		childNode->setBranchesSize(0);

		float acc = accuracy->calculateAccuracy(decisionTreeHead, validDataHead, noOfValidData);

		tempCurNode->deepCopy(childNode);

		if (acc >= bestPruneNodeStruct->validAcc)
		{
			bestPruneNodeStruct->validAcc = acc;
			bestPruneNodeStruct->nodeToPrune = childNode;
		}
	}
}

void pruneNode(treeNode * treeNodeToPrune)
{
	for (int i = 0; i < treeNodeToPrune->getNoOfBranches(); i++)
	{
		treeNode * child = treeNodeToPrune->getChild(i);
		pruneNode(child);
		
		delete child;
		child = NULL;
		child = 0;
	}

	treeNodeToPrune->setAttrNode(NULL);
	treeNodeToPrune->setAttributeIndex(-1);
	treeNodeToPrune->setChildren(NULL);
	treeNodeToPrune->setCondition(NULL);
	treeNodeToPrune->setLogicalCondition(NULL);
	treeNodeToPrune->setBranchesSize(0);
}

void postPruningNodes(treeNode * decisionTreeHead, Accuracy * accuracy, dataRow * validDataHead, int noOfValidData)
{
	float curAccuracy = accuracy->calculateAccuracy(decisionTreeHead, validDataHead, noOfValidData);
	float newAccuracy = curAccuracy;

	int noOfNodesInTree = 0;
	getNumberOfNodes(decisionTreeHead, noOfNodesInTree);
	accuracy->calculateTrainAccuracy(noOfNodesInTree, decisionTreeHead);
	accuracy->calculateTestAccuracy(noOfNodesInTree, decisionTreeHead);
	accuracy->calculateValidAccuracy(noOfNodesInTree, decisionTreeHead);

	bestPruneNode * bestPruneNodeStruct = new bestPruneNode();

	int cntr = 0;
	while (1)
	{
		//cout << "Pruning = " << cntr++ << "\n";

		// Get Prune Node
		treeNode * tempCurNode = new treeNode();
		bestPruneNodeStruct->nodeToPrune = NULL;
		bestPruneNodeStruct->validAcc = curAccuracy;

		getBestTreeNodeToPrune(decisionTreeHead, decisionTreeHead, accuracy, validDataHead, noOfValidData, noOfNodesInTree, tempCurNode, bestPruneNodeStruct);
		
		delete tempCurNode;
		tempCurNode = 0;
		tempCurNode = NULL;

		// Prune Node
		newAccuracy = bestPruneNodeStruct->validAcc;
		if (curAccuracy <= newAccuracy && bestPruneNodeStruct->nodeToPrune != NULL)
		{
			//prune node
			pruneNode(bestPruneNodeStruct->nodeToPrune);

			noOfNodesInTree = 0;
			getNumberOfNodes(decisionTreeHead, noOfNodesInTree);

			// Calculate Accuracy
			accuracy->calculateTrainAccuracy(noOfNodesInTree, decisionTreeHead);
			accuracy->calculateTestAccuracy(noOfNodesInTree, decisionTreeHead);
			accuracy->calculateValidAccuracy(noOfNodesInTree, decisionTreeHead);
		}
		else
			break;

		curAccuracy = newAccuracy;
	}
}

struct AttrSplitMultTimes
{
	int cnt;
	string attrName;
	string attrIndex;
	int attrId;
	int condition[4300];
	int condCnt;

	AttrSplitMultTimes * next;

	AttrSplitMultTimes()
	{
		cnt = 0;
		attrName = "";
		attrIndex = "";
		attrId = -1;
		condCnt = 0;
		next = NULL;
	}
};

void countAttributeSplitMulTimes(treeNode * decisionTreeNode, AttrSplitMultTimes ** attrSplit)
{
	AttrSplitMultTimes * ltemp = *attrSplit;
	for (treeNode * node = decisionTreeNode; node->getParentNode() != NULL; node = node->getParentNode())
	{
		if (node->getAttrNode()->getCategory() == continuos)
		{
			AttrSplitMultTimes * lchosen = ltemp;
			for (; lchosen->next != NULL; lchosen = lchosen->next)
			{
				if (lchosen->attrId == node->getAttributeIndex()) break;
			}

			lchosen->cnt++;
			lchosen->condition[lchosen->condCnt++] = node->getCondition(0);
			if (lchosen->attrName == "")
			{
				lchosen->attrName = node->getAttrNode()->getAttributeName();
				lchosen->attrId = node->getAttributeIndex();
				lchosen->attrIndex = node->getAttrNode()->getAttributeId();
				lchosen->next = new AttrSplitMultTimes();
			}
		}
	}
}

void displayAttributesSplitMultipleTimes(ofstream * outfile, AttrSplitMultTimes ** attrSplit, int branchNo)
{
	if ((*attrSplit)->attrName == "") return;

	(*outfile) << to_string(branchNo) << "\n";

	for (AttrSplitMultTimes * ltemp = *attrSplit; ltemp->next->next != NULL; ltemp = ltemp->next)
	{
		(*outfile) << ltemp->attrIndex << " " << ltemp->attrName << ": Number of Occ = " << to_string(ltemp->cnt) << ": ";
		for (int i = 0; i < ltemp->condCnt; i++)
		{
			(*outfile) << ltemp->condition[i] << ", ";
		}
		(*outfile) << "\n";
	}

	(*outfile) << "\n";
}

void countAndDispAttributesSplitMultipleTimes(treeNode * decisionTreeNode, ofstream * outfile, int & branchNo, AttrSplitMultTimes ** attrSplit)
{
	if (decisionTreeNode->getNoOfBranches() == 0) return;

	bool areAllChildLeaf = true;
	int nob1 = decisionTreeNode->getNoOfBranches();
	for (int i = 0; i < nob1; i++)
	{
		if (decisionTreeNode->getChild(i)->getNoOfBranches() > 0)
		{
			areAllChildLeaf = false;
			break;
		}
	}

	if (areAllChildLeaf)		// reached a leaf node's parent
	{
		countAttributeSplitMulTimes(decisionTreeNode->getParentNode(), attrSplit);
		displayAttributesSplitMultipleTimes(outfile, attrSplit, branchNo);
		branchNo++;

		for (AttrSplitMultTimes * attrSplitTemp = *attrSplit; attrSplitTemp != NULL; )
		{
			AttrSplitMultTimes * temp = attrSplitTemp->next;
			delete attrSplitTemp;
			attrSplitTemp = 0;
			attrSplitTemp = NULL;
			attrSplitTemp = temp;
		}

		*attrSplit = NULL;
		*attrSplit = 0;
		*attrSplit = new AttrSplitMultTimes();
	}

	int nob = decisionTreeNode->getNoOfBranches();
	for (int i = 0; i < nob; i++)
	{
		countAndDispAttributesSplitMultipleTimes(decisionTreeNode->getChild(i), outfile, branchNo, attrSplit);
	}
}

int main(int argc, char * argv[])
{
	char * trainfilename = argv[1];
	char * validfilename = argv[2];
	char * testfilename = argv[3];
	char * partnum = argv[4];
	char * outfile = argv[5];

	//cout << trainfilename << "\n" << validfilename << "\n" << testfilename << "\n" << partnum << "\n" << outfile << "\n";

	// Populate Data
	dataRow * trainDataHead = new dataRow();
	dataRow * validDataHead = new dataRow();
	dataRow * testDataHead = new dataRow();
	attribute * attrsHead = new attribute();
	int noOfTrainingData = fillData(trainDataHead, trainfilename, attrsHead);
	int noOfTestData = fillData(testDataHead, testfilename);
	int noOfValidData = fillData(validDataHead, validfilename);

	// Preprocess Data
	preprocessData(trainDataHead, attrsHead, noOfTrainingData, true, partnum);
	preprocessData(validDataHead, attrsHead, noOfTestData, false, partnum);
	preprocessData(testDataHead, attrsHead, noOfValidData, false, partnum);

	// Initialize Accuracy
	Accuracy * accuracy = new Accuracy();
	accuracy->setPartNum(partnum);
	accuracy->setTrainData(trainDataHead);
	accuracy->setNoOfTrainData(noOfTrainingData);
	accuracy->setTestData(testDataHead);
	accuracy->setNoOfTestData(noOfTestData);
	accuracy->setValidData(validDataHead);
	accuracy->setNoOfValidData(noOfValidData);

	// Construct Tree
	//cerr << "Constructing Tree\n";
	treeNode * decisionTreeHead = new treeNode;
	treeNode * decisionTreeTail = decisionTreeHead;
	decisionTreeTail->storeTrainData(trainDataHead, noOfTrainingData);
	decisionTreeTail->countNoOfTrueFalseDecisions();

	attribute * attrsTemp = attrsHead;
	int noOfAttrs = trainDataHead->getNoOfAttributes()-1;		// -1 is to minus the y attribute
	attribute ** attrsArr = (attribute **)malloc(sizeof(attribute *) * noOfAttrs);
	for (int i = 0; i < noOfAttrs; i++, attrsTemp = attrsTemp->getNext())
		attrsArr[i] = attrsTemp;

	int constructingNode = 0;
	constructDecisionTree(attrsArr, noOfAttrs, decisionTreeHead, decisionTreeTail, accuracy, constructingNode);

	if (strcmp(partnum, "b") == 0)		// Post Pruning
	{
		postPruningNodes(decisionTreeHead, accuracy, validDataHead, noOfValidData);
	}

	// Print Accuracy
	AccuracyData * accData = accuracy->getAccuracyData();
	int cntr = 0;
	for (AccuracyData * ltemp = accData; ltemp->next != NULL; ltemp = ltemp->next, cntr++);

	int * retnode = (int *)malloc(sizeof(int) * cntr);
	float * rettrainacc = (float *)malloc(sizeof(float) * cntr);
	float * rettestacc = (float *)malloc(sizeof(float) * cntr);
	float * retvalacc = (float *)malloc(sizeof(float) * cntr);

	cntr = 0;
	for (AccuracyData * ltemp = accData; ltemp->next != NULL; ltemp = ltemp->next, cntr++)
	{
		retnode[cntr] = ltemp->nodeNumber;
		rettrainacc[cntr] = ltemp->trainAccuracy;
		retvalacc[cntr] = ltemp->validAccuracy;
		rettestacc[cntr] = ltemp->testAccuracy;
	}

	ofstream outf;
	outf.open(outfile, ios::out);
	for (int i = 0; i < cntr; i++)
		outf << to_string(retnode[i]) << "\n";
	outf << "\n";
	for (int i = 0; i < cntr; i++)
	{
		outf << to_string(rettrainacc[i]) << "\n";
		if (i == cntr - 1)
			cout << "Training Accuracy = " << to_string(rettrainacc[i]) << "\n";
	}
	outf << "\n";
	for (int i = 0; i < cntr; i++)
	{
		outf << to_string(retvalacc[i]) << "\n";
		if (i == cntr - 1)
			cout << "Validation Accuracy = " << to_string(retvalacc[i]) << "\n";
	}
	outf << "\n";
	for (int i = 0; i < cntr; i++)
	{
		outf << to_string(rettestacc[i]) << "\n";
		if (i == cntr - 1)
			cout << "Testing Accuracy = " << to_string(rettestacc[i]) << "\n";
	}
	outf.close();

	ofstream outf1;
	string of(outfile);
	string outfile1 = of.substr(0, of.find_last_of("\\/") + 1) + "partc_attr";
	outf1.open(outfile1, ios::out);

	int branchNo = 1;
	AttrSplitMultTimes * attrSplit = new AttrSplitMultTimes();
	countAndDispAttributesSplitMultipleTimes(decisionTreeHead, &outf1, branchNo, &attrSplit);

	outf1.close();

	//decisionTreeHead->print(1);

	//int i;
	//cin >> i;
	return 0;
}