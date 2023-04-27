from sklearn import svm
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix


def SVMOwnDataSet(locations, filename, partOfData):
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    bestModel, score = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    if( partOfData == 1): return score

    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    return score


def SVMAgainstOtherDatasets(locations, filename, filenameTests, partOfData):

    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData)
    
    bestModel, _ = bestModelSVM(trainingSamples, trainingLabels)
    
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