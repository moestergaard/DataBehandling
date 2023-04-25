import numpy as np
from MatrixManipulation import changeMatrice

#
# Makes an array consisting of the distinct BSSID.
#
def extractDistinctBSSIDAndNumberOfDataPoints(filename, locations, distinctBSSIDTraining = []):
    if(distinctBSSIDTraining != []):
        distinctBSSID = [0] * len(distinctBSSIDTraining)
    else: distinctBSSID = []
    dataPoints = 0
    dataPointIncluded = False

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("BSSID"):
                if dataPointIncluded:
                    bssid = line.split(": ")[1].strip()
                    alreadyIncludedInDistinctBSSID = distinctBSSID.__contains__(bssid)
                    if not alreadyIncludedInDistinctBSSID:
                        if(distinctBSSIDTraining != []):
                            for i in range(0, len(distinctBSSIDTraining)):
                                if bssid.__eq__(distinctBSSIDTraining[i]):
                                    distinctBSSID[i] = bssid
                        else: distinctBSSID.append(bssid)
            if line.__contains__("Scanning"):
                dataPointIncluded = False
                for i in range(0, len(locations)):
                    if line.__contains__(locations[i]):    
                        dataPoints += 1
                        dataPointIncluded = True

    return distinctBSSID, dataPoints

#
# Makes the following matrices
#   For training: m is the number of training samples, and n is the number of features (distinct BSSID)
#   A mxn matrice
#   A mx1 matrice:  The labels corresponding to training samples.
#  
#   For testing: k is the number of test samples, and n is the same as before.
#   A kxn matrice
#   A kx1 matrice:  The labels corresponding to test samples.
#
def extractData(filename, distinctBSSID, samples, locations, ratio):

    numberOfFeatures = len(distinctBSSID)
    numberOfTestSamples = np.ceil(samples/5).astype(int)
    numberOfTrainingSamples = samples - numberOfTestSamples

    trainingSamples = np.zeros((numberOfTrainingSamples, numberOfFeatures))
    labelsTrainingSamples = np.zeros((numberOfTrainingSamples, ))
    
    testSamples = np.zeros((numberOfTestSamples, numberOfFeatures))
    labelsTestSamples = np.zeros((numberOfTestSamples, ))

    twentyPercentTestData = 4
    indexTestSample = -1
    indexTrainingSample = -1
    
    dataPointIncluded = False

    currentBSSID = ""
    

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"):
                twentyPercentTestData = np.mod(twentyPercentTestData + 1, 5)
            if line.__contains__("BSSID"):
                if dataPointIncluded:
                    currentBSSID = line.split(": ")[1].split()
            if line.__contains__("Scanning"):
                dataPointIncluded = False
                location = line.split(": ")[1].strip()
                for i in range(0, len(locations)):
                    if locations[i].__eq__(location):
                        if twentyPercentTestData == 4:
                            indexTestSample += 1
                            labelsTestSamples[indexTestSample] = i
                            dataPointIncluded = True
                        else:
                            indexTrainingSample += 1
                            labelsTrainingSamples[indexTrainingSample] = i
                            dataPointIncluded = True
                if not dataPointIncluded:
                    if twentyPercentTestData == 0: twentyPercentTestData = 4
                    else: twentyPercentTestData -= 1
            if line.__contains__("ResultLevel"):
                if dataPointIncluded:
                    resultLevel = line.split(": ")[1].split()
                    if twentyPercentTestData == 4:
                        testSamples = changeMatrice(testSamples, indexTestSample, distinctBSSID, currentBSSID[0], resultLevel[0])
                    else: trainingSamples = changeMatrice(trainingSamples, indexTrainingSample, distinctBSSID, currentBSSID[0], resultLevel[0])

    tmpRatio = 0
    tmpTrainingSamples = trainingSamples
    
    return trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples

def extractDataCombined(filename, distinctBSSID, numberOfSamples, locations):

    samples = np.zeros((numberOfSamples, len(distinctBSSID)))
    labels = np.zeros((numberOfSamples, ))
    
    index = 0

    currentBSSID = ""
    dataPointIncluded = False

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"): 
                index += 1
            if line.__contains__("BSSID"):
                if dataPointIncluded:
                    currentBSSID = line.split(": ")[1].split()
            if line.__contains__("Scanning"):
                dataPointIncluded = False
                location = line.split(": ")[1].strip()
                for i in range(0, len(locations)):
                    if locations[i].__eq__(location): 
                        dataPointIncluded = True
                        labels[index] = i
                if not dataPointIncluded:
                    index -= 1
            if line.__contains__("ResultLevel"):
                if dataPointIncluded:
                    resultLevel = line.split(": ")[1].split()
                    samples = changeMatrice(samples, index, distinctBSSID, currentBSSID[0], resultLevel[0])

    return samples, labels


def extractDataFromMultipleFiles(filenameTests, locations, distinctBSSID):
    listOfTestSamples = []
    listOfTestLabels = []
    for filenameTest in filenameTests:
        distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, locations, distinctBSSID)
        testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSID, dataPointsTest, locations)
        
        listOfTestSamples.append(testSamples)
        listOfTestLabels.append(testLabels)
        
    allTestSamples = np.concatenate(listOfTestSamples)
    allTestLabels = np.concatenate(listOfTestLabels)
    
    return allTestSamples, allTestLabels





