import numpy as np
from ExtractData import extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataFromMultipleFiles, getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from NeuralNetwork import trainingModelNN, getPredictedLabelsNN, testingNN
from MatrixManipulation import randomSplitSamplesAndLabels, deterministicSplitMatrix

def NNOwnDataSet(locations, filename, partOfData, bias):
    
    # distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    # samples, labels = extractData(filename, distinctBSSID, dataPoints, locations)
    
    # trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    wh, bh, wo, bo, error_cost_list, error_cost, percentageSure, accuracy = bestModelNN(trainingSamplesOverall, trainingLabelsOverall, bias, numberOfClasses=len(locations))
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    if partOfData == 1: return accuracy, fiveFractile
    
    predictedLabels, percentageSure = getPredictedLabelsNN(testSamplesOverall, testLabelsOverall, wh, bh, wo, bo, error_cost_list, error_cost)
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    return accuracy, fiveFractile
    

def NNAgainstOtherDatasets(locations, filename, filenameTests, partOfData, bias):

    # distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    # trainingSamples, trainingLabels = extractData(filename, distinctBSSID, dataPoints, locations)
    
    # testSamples, testLabels = extractDataFromMultipleFiles(filenameTests, locations, distinctBSSID)
    
    # trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(trainingSamples, trainingLabels, partOfData)
    
    # testSamples = np.concatenate((testSamples, testSamplesOverall))
    # testLabels = np.concatenate((testLabels, testLabelsOverall))
    
    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData)
    
    wh, bh, wo, bo, error_cost_list, error_cost, _, _ = bestModelNN(trainingSamples, trainingLabels, bias, numberOfClasses=len(locations))
    
    predictedLabels, percentageSure = getPredictedLabelsNN(testSamplesOverall, testLabelsOverall, wh, bh, wo, bo, error_cost_list, error_cost)
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    return accuracy, fiveFractile
    
    # print("************")
    
    distin = ""
    for i in range(len(distinctBSSID)):
        distin += "\"" + distinctBSSID[i] + "\", "
    # print(distin)
    print("************")
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, distinctBSSID)

    

    trainingSamples, trainingLabels = extractData(filename, distinctBSSID, dataPoints, locations)
    # print("labels training: ", trainingLabels)

    testSamples, testLabels = extractData(filenameTest, distinctBSSIDTest, dataPointsTest, locations)
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


def bestModelNN(samples, labels, bias, numberOfClasses):
    bestAccuracy = float('-inf')
    bestwh = None
    bestbh = None
    bestwo = None
    bestbo = None
    bestError_cost_list = None
    bestError_cost = None
    bestPercentageSure = None
    
    for i in range(1,6):
        trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(samples, labels, 1/5, i)
        wh, bh, wo, bo, error_cost_list, error_cost = trainingModelNN(trainingSamples, trainingLabels, bias, numberOfClasses)
        
        predictedLabels, percentageSure = getPredictedLabelsNN(testSamples, testLabels, wh, bh, wo, bo, error_cost_list, error_cost)
        accuracy = testingNN(testLabels, predictedLabels)
        
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestwh = wh
            bestbh = bh
            bestwo = wo
            bestbo = bo
            bestError_cost_list = error_cost_list
            bestError_cost = error_cost
            bestPercentageSure = percentageSure
    
    return bestwh, bestbh, bestwo, bestbo, bestError_cost_list, bestError_cost, bestPercentageSure, bestAccuracy
    

def DataSet(locations, filename):
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    # print("Training Samples: ", trainingSamples)
    # print("Labels Training Samples: ", labelsTrainingSamples)
    # print("Test Samples: ", testSamples)
    # print("Labels Test Samples: ", labelsTestSamples)
    
    wh, bh, wo, bo, error_cost_list, error_cost = calculationsNN(trainingSamples, labelsTrainingSamples, testSamples, numberOfClasses = len(locations))
    
    print()
    print("********************************************************************************************")
    print()
    print(filename)
    
    accuracyNN(testSamples, labelsTestSamples, wh, bh, wo, bo, error_cost_list, error_cost)
