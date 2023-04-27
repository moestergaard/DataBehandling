import numpy as np
from sklearn import svm
from ExtractData import extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataFromMultipleFiles
from MatrixManipulation import randomSplitSamplesAndLabels, deterministicSplitMatrix


def SVMAgainstOtherDatasets(locations, filename, filenameTests, partOfData):

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    trainingSamples, trainingLabels = extractData(filename, distinctBSSID, dataPoints, locations)
    
    testSamples, testLabels = extractDataFromMultipleFiles(filenameTests, locations, distinctBSSID)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(trainingSamples, trainingLabels, partOfData)
    
    testSamples = np.concatenate((testSamples, testSamplesOverall))
    testLabels = np.concatenate((testLabels, testLabelsOverall))
    
    bestModel, _ = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    score = bestModel.score(testSamples, testLabels)    
    return score
        

def SVMOwnDataSet(locations, filename, partOfData):
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    samples, labels = extractData(filename, distinctBSSID, dataPoints, locations)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    bestModel, score = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    if( partOfData == 1): return score

    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    return score
   
        
def bestModelSVM(samples, labels):
    bestScore = float('-inf')
    bestModel = None
    
    for i in range(1,6):
        trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(samples, labels, 1/5, i)
        model = fitModel(trainingSamples, trainingLabels)
        
        score = model.score(testSamples, testLabels)
        
        if score > bestScore:
            bestScore = score
            bestModel = model
    
    return bestModel, bestScore
        
        
def fitModel(trainingSamples, labelsTrainingSamples):
    clf = svm.SVC(cache_size = 1000, class_weight='balanced')
    clf.fit(trainingSamples, labelsTrainingSamples)

    return clf