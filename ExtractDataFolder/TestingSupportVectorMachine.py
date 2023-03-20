import numpy as np
from sklearn import svm
from ExtractData import changeDataFile, extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataSeparateFloors, extractDataCombined
from SupportVectorMachine import calculationsSVM, accuracySVM

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    groundFloor = ["Stue, Køkken"]
    firstFloor = ["Kontor"]

    bigDataSetSVM(locations)
    smallDataSetTestedAgainstBigDataSetSVM(locations)

    bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations)
    smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations)
    
    # changeDataFile(filename)
    
def bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations):
    filename = 'WifiData2303141637Modified.txt'
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)

    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)
    trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor = extractDataSeparateFloors(trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples)

    predictionSVM = calculationsSVM(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** BigDataSetSVMSeparateFloors ***")
    accuracySVM(labelsTestFloors, predictionSVM)

    predictionSVM = calculationsSVM(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** BigDataSetSVMSeparateFloors Ground Floor ***")
    accuracySVM(labelsTestGroundFloor, predictionSVM)

def smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations):
    filename = 'WifiData2-2303172344.txt'
    filenameTest = 'WifiData2303141637Modified.txt'

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest)

    

    trainingSamples, trainingLabels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)
    
    testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSIDTest, dataPointsTest, locations)

    #
    # Removes all new BSSID that were not present at origianl syncronising
    #
    for i in range(0, len(distinctBSSIDTest)):
        for i in range(0, len(distinctBSSIDTest)):
            if not distinctBSSID.__contains__(distinctBSSIDTest[i]):
                testSamples = np.delete(testSamples, i, 1)
                distinctBSSIDTest.remove(distinctBSSIDTest[i])
                break

    #
    # Add rows of zeroes to testSample for all BSSID from syncroized data not present in tests
    #
    for i in range(0, len(distinctBSSID)):
        if not distinctBSSIDTest.__contains__(distinctBSSID[i]):
            distinctBSSIDTest.append(distinctBSSID[i])
            testSamples = np.append(testSamples, np.zeros((testSamples.shape[0],1)), 1)

    trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor = extractDataSeparateFloors(trainingSamples, trainingLabels, testSamples, testLabels)
                
    predictionSVM = calculationsSVM(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** SmallDataSetTestedAgainstBigDataSetSVMSeparateFloors ***")
    accuracySVM(labelsTestFloors, predictionSVM)

    predictionSVM = calculationsSVM(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** SmallDataSetTestedAgainstBigDataSetSVMSeparateFloors Ground Floor ***")
    accuracySVM(labelsTestGroundFloor, predictionSVM)


def smallDataSetTestedAgainstBigDataSetSVM(locations):
    filename = 'WifiData2-2303172344.txt'
    filenameTest = 'WifiData2303141637Modified.txt'

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest)

    

    trainingSamples, trainingLabels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)

    testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSIDTest, dataPointsTest, locations)

    #
    # Removes all new BSSID that were not present at origianl syncronising
    #
    for i in range(0, len(distinctBSSIDTest)):
        for i in range(0, len(distinctBSSIDTest)):
            if not distinctBSSID.__contains__(distinctBSSIDTest[i]):
                testSamples = np.delete(testSamples, i, 1)
                distinctBSSIDTest.remove(distinctBSSIDTest[i])
                break

    #
    # Add rows of zeroes to testSample for all BSSID from syncroized data not present in tests
    #
    for i in range(0, len(distinctBSSID)):
        if not distinctBSSIDTest.__contains__(distinctBSSID[i]):
            distinctBSSIDTest.append(distinctBSSID[i])
            testSamples = np.append(testSamples, np.zeros((testSamples.shape[0],1)), 1)
                
    predictionSVM = calculationsSVM(trainingSamples, trainingLabels, testSamples)

    print("*** SmallDataSetTestedAgainstBigDataSetSVM ***")
    accuracySVM(testLabels, predictionSVM)


def bigDataSetSVM(locations):
    filename = 'WifiData2303141637Modified.txt'
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    predictionSVM = calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples)
    print("*** BigDataSetSVM ***")
    accuracySVM(labelsTestSamples, predictionSVM)

if __name__ == '__main__':
    main()