import numpy as np
from sklearn import svm
from ExtractData import changeDataFile, extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataSeparateFloors, extractDataCombined
from SupportVectorMachine import calculationsSVM, accuracySVM

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    groundFloor = ["Stue, Køkken"]
    firstFloor = ["Kontor"]

    filenameBigDataSet = "BigDataSet.txt"
    filename5Minute = "5MinuteDataSet.txt"
    filename10Minute = "10MinuteDataSet.txt"
    filenameUpdated = "WifiData230324.txt"

    DataSet(locations, filenameUpdated)
    DataSet(locations, filenameBigDataSet)
    DataSet(locations, filename5Minute)
    DataSet(locations, filename10Minute)
    
    smallDataSetTestedAgainstBigDataSet(locations, filenameUpdated, filename5Minute)
    smallDataSetTestedAgainstBigDataSet(locations, filenameUpdated, filename10Minute)
    smallDataSetTestedAgainstBigDataSet(locations, filename5Minute, filenameBigDataSet)
    smallDataSetTestedAgainstBigDataSet(locations, filename10Minute, filenameBigDataSet)
    
    # bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations)
    # smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations)
    
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


def smallDataSetTestedAgainstBigDataSet(locations, filename, filenameTest):

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, distinctBSSID)

    

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

    print()
    print("********************************************************************************************")
    print()
    print(f"{filename} tested against {filenameTest}")
    
    accuracySVM(testLabels, predictionSVM)


def DataSet(locations, filename):
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    predictionSVM = calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples)
    
    print()
    print("********************************************************************************************")
    print()
    print(filename)

    accuracySVM(labelsTestSamples, predictionSVM)

if __name__ == '__main__':
    main()