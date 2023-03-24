import numpy as np
from ExtractData import changeDataFile, extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataSeparateFloors, extractDataCombined
from NeuralNetwork import calculationsNN, accuracyNN

def main():
    locations = ["Kontor", "Stue", "Køkken", "Entre"]
    groundFloor = ["Stue, Køkken"]
    firstFloor = ["Kontor"]
    
    filenameBigDataSet = "BigDataSet.txt"
    filename5Minute = "5MinuteDataSet.txt"
    filename10Minute = "10MinuteDataSet.txt"
    filenameNoSpeakers = "NoSpeakers.txt"
    filenameUpdated = "WifiData230324.txt"

    DataSet(locations, filename5Minute)
    DataSet(locations, filename10Minute)
    DataSet(locations, filenameUpdated)

    
    smallDataSetTestedAgainstBigDataSet(locations, filenameUpdated, filenameBigDataSet)
    smallDataSetTestedAgainstBigDataSet(locations, filename5Minute, filenameBigDataSet)
    smallDataSetTestedAgainstBigDataSet(locations, filename10Minute, filenameBigDataSet)
    smallDataSetTestedAgainstBigDataSet(locations, filenameBigDataSet, filename10Minute)
    
    # smallDataSetTestedAgainstBigDataSet(locations, filenameBigDataSet, filenameNoSpeakers)
    # smallDataSetTestedAgainstBigDataSet(locations, filename5Minute, filenameNoSpeakers)
    # smallDataSetTestedAgainstBigDataSet(locations, filename10Minute, filenameNoSpeakers)

    # bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations)
    # smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations)
    
    # filename = "WifiData2303222123.txt"
    
    # changeDataFile(filename)
    
def bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations, filename):
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)

    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)
    trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor = extractDataSeparateFloors(trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples)

    wh, bh, wo, bo, error_cost = calculationsNN(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** BigDataSetSeparateFloors ***")
    accuracyNN(testSamplesFloors, labelsTestFloors, wh, bh, wo, bo, error_cost)

    wh, bh, wo, bo, error_cost = calculationsNN(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** BigDataSetSeparateFloors Ground Floor ***")
    accuracyNN(testSamplesGroundFloor, labelsTestGroundFloor, wh, bh, wo, bo, error_cost)

def smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations, filename, filenameTest):

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
                
    wh, bh, wo, bo, error_cost = calculationsNN(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** SmallDataSetTestedAgainstBigDataSetSeparateFloors ***")
    accuracyNN(testSamplesFloors, labelsTestFloors, wh, bh, wo, bo, error_cost)

    wh, bh, wo, bo, error_cost = calculationsNN(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** SmallDataSetTestedAgainstBigDataSetSeparateFloors Ground Floor ***")
    accuracyNN(testSamplesGroundFloor, labelsTestGroundFloor, wh, bh, wo, bo, error_cost)


def smallDataSetTestedAgainstBigDataSet(locations, filename, filenameTest):

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    # print("************")
    
    distin = ""
    for i in range(len(distinctBSSID)):
        distin += "\"" + distinctBSSID[i] + "\", "
    print(distin)
    print("************")
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, distinctBSSID)

    

    trainingSamples, trainingLabels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)
    # print("labels training: ", trainingLabels)

    testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSIDTest, dataPointsTest, locations)
    # print("labels test: ", testLabels)

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
                
    wh, bh, wo, bo, error_cost_list, error_cost = calculationsNN(trainingSamples, trainingLabels, testSamples)

    print()
    print("********************************************************************************************")
    print()
    print(f"{filename} tested against {filenameTest}")
    
    accuracyNN(testSamples, testLabels, wh, bh, wo, bo, error_cost_list, error_cost)


def DataSet(locations, filename):
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    print("Training Samples: ", trainingSamples)
    print("Labels Training Samples: ", labelsTrainingSamples)
    print("Test Samples: ", testSamples)
    print("Labels Test Samples: ", labelsTestSamples)
    
    wh, bh, wo, bo, error_cost_list, error_cost = calculationsNN(trainingSamples, labelsTrainingSamples, testSamples)
    
    print()
    print("********************************************************************************************")
    print()
    print(filename)
    
    accuracyNN(testSamples, labelsTestSamples, wh, bh, wo, bo, error_cost_list, error_cost)

if __name__ == '__main__':
    main()