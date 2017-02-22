import csv
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy

numEntries = -1
numFeatures = -1

# features = [[x1, x2, ...], [y1, y2, ...], ...]
features = []
featureSums = []
featureMeans = []
featureStdDevs = []
classes = []


def RMSE(expVals, thVals):
    squaredError = 0
    for i in range(len(expVals)):
        squaredError += (expVals[i]-thVals[i])**2
    return math.sqrt(squaredError/len(expVals))


# Read the data from csv
with open('x06Simple.csv', 'rt') as csvFile:
    csvReader = csv.reader(csvFile)
    for row in csvReader:
        # Ignore the first labels row
        if numEntries is -1:
            numEntries = 0
            continue

        numEntries = numEntries+1
        if numFeatures is -1:
            numFeatures = len(row)-1
            features = [[] for i in range(numFeatures)]
            featureSums = [0]*numFeatures
            featureMeans = [0]*numFeatures
            featureStdDevs = [0]*numFeatures

        i = 0
        for item in row:
            item = float(item)
            if i is not 0:
                featureSums[i-1] += item
                features[i-1].append(item)
            else:
                classes.append(item)
            i = i+1

# Compute the mean and standard deviation
for i in range(numFeatures):
    featureMeans[i] = featureSums[i]/numEntries
    sum = 0.0
    for item in features[i]:
        sum += (item-featureMeans[i])**2
    average = sum/numEntries
    featureStdDevs[i] = math.sqrt(average)

# Standardize the data
featuresStd = copy.deepcopy(features)
for i in range(numFeatures):
    for j in range(numEntries):
        featuresStd[i][j] = (features[i][j]-featureMeans[i])/featureStdDevs[i]

numEntriesTrain = int(math.floor(numEntries * (2.0/3.0)))
numEntriesTest = int(math.ceil(numEntries * (1.0/3.0)))

featuresTest = [[] for i in features]
featuresStdTest = [[] for i in features]

random.seed(0)

# Randomize the data and select training and testing data
index_shuf = range(numEntries-1)
random.shuffle(index_shuf)
for j in range(len(featuresStd)):
    featuresStdTest[j] = [featuresStd[j][i] for i in index_shuf[numEntriesTrain:]]
    featuresTest[j] = [features[j][i] for i in index_shuf[numEntriesTrain:]]
    featuresStd[j] = [featuresStd[j][i] for i in index_shuf[:numEntriesTrain]]
    features[j] = [features[j][i] for i in index_shuf[:numEntriesTrain]]

featuresStd = [[1]*numEntriesTrain] + featuresStd
featuresStdTest = [[1]*(numEntriesTest-1)] + featuresStdTest
numFeatures += 1

random.seed(0)
theta = [(random.random()*2 - 1) for i in range(numFeatures-1)]
maxIt = 1000000
minDeltaRMSE = 2**-52  # Value of eps in matlab
it = 0
rmse = 0
deltaRMSE = 1
learningRate = 0.1

X = np.asarray(featuresStd[:-1])
Y = np.array(features[len(features)-1])

testX = np.asarray(featuresStdTest[:-1])
testY = np.array(featuresTest[len(featuresTest)-1])
theta = np.asarray(theta)
m = Y.size
rmseLog = []
rmseTestLog = []
while it < maxIt and deltaRMSE > minDeltaRMSE:
    beta = deepcopy(theta)
    for feature in range(len(theta)):
        sum = 0
        for entry in range(m):
            sum += (X.T[entry].dot(theta) - Y[entry])*X[feature][entry]
        avg = sum/m
        avg = learningRate*avg
        beta[feature] = theta[feature] - avg
    theta = beta

    results = X.T.dot(theta)
    rmseNew = RMSE(results, Y)

    resultsTest = testX.T.dot(theta)
    rmseTest = RMSE(resultsTest, testY)
    rmseTestLog.append(rmseTest)

    deltaRMSE = abs(rmse-rmseNew)
    rmse = rmseNew
    rmseLog.append(rmse)
    it += 1

print("Final Model:  Y = "
      + str(int(theta[0]))
      + " + "
      + " + ".join([str(int(theta[i]))
                    + "*X"
                    + str(i) for i in range(1, len(theta))]))

print("Final Training RMSE = " + str(rmseLog[len(rmseLog)-1]))
print("Final Testing RMSE = " + str(rmseTestLog[len(rmseTestLog)-1]))

l1, = plt.plot(rmseTestLog, color="b", label="RMSE of Test Data")
l2, = plt.plot(rmseLog, color="r", label="RMSE of Training Data")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.show()
