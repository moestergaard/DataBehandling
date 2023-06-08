import numpy as np
from MatrixManipulation import changeMatrice, randomSplitSamplesAndLabels

def getSamplesAndLabelsFromOneFile(locations, filename, partOfData):
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, filename)
    samples, labels = extractData(locations, filename, distinctBSSID, dataPoints)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    return trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall


def getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData, testNotARoom = False):
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, filename)
    trainingSamples, trainingLabels = extractData(locations, filename, distinctBSSID, dataPoints)
    
    testSamplesOverall, testLabelsOverall = extractDataFromMultipleFiles(locations, filenameTests, distinctBSSID)
    
    newTrainingSamples, newTestSamples, newTrainingLabels, newTestLabels = randomSplitSamplesAndLabels(trainingSamples, trainingLabels, partOfData)
    
    if not testNotARoom:
        testSamplesOverall = np.concatenate((newTestSamples, testSamplesOverall))
        testLabelsOverall = np.concatenate((newTestLabels, testLabelsOverall))
    
    return newTrainingSamples, testSamplesOverall, newTrainingLabels, testLabelsOverall


def extractDataFromMultipleFiles(locations, filenameTests, distinctBSSID):
    listOfTestSamples = []
    listOfTestLabels = []
    for filenameTest in filenameTests:
        _ , dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(locations, filenameTest, distinctBSSID)
        testSamples, testLabels = extractData(locations, filenameTest, distinctBSSID, dataPointsTest)
        
        listOfTestSamples.append(testSamples)
        listOfTestLabels.append(testLabels)
        
    allTestSamples = np.concatenate(listOfTestSamples)
    allTestLabels = np.concatenate(listOfTestLabels)
    
    return allTestSamples, allTestLabels


def extractDistinctBSSIDAndNumberOfDataPoints(locations, filename, distinctBSSIDTraining = []):
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

def extractData(locations, filename, distinctBSSID, numberOfSamples):

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