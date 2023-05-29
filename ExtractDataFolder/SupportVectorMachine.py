from sklearn import svm
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix
from libsvm.svmutil import svm_problem, svm_parameter, svm_train
import numpy as np


def SVMOwnDataSet(locations, filename, partOfData):
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    bestModel, score = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    if( partOfData == 1): return score

    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    return score


def SVMAgainstOtherDatasets(locations, filename, filenameTests, partOfData):

    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData)
    
    model = fitModel(trainingSamples, trainingLabels)
    score = model.score(testSamplesOverall, testLabelsOverall)    
    
    # bestModel, _ = bestModelSVM(trainingSamples, trainingLabels)
    #score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    
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
    # clf = svm.LinearSVC(dual=False, class_weight='balanced', max_iter=100000)
    clf = svm.SVC(cache_size = 1000, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(trainingSamples, labelsTrainingSamples)

    return clf

def fitModel3(trainingSamples, labelsTrainingSamples):
    problem = svm_problem(labelsTrainingSamples, trainingSamples)
    param = svm_parameter('-t 2 -s 0 -b 1 -c 1 -w1 1 -w-1 1')
    model = svm_train(problem, param)
    return model

def fitModel2(trainingSamples, labelsTrainingSamples):
    labels = list(labelsTrainingSamples)
    # features = list(trainingSamples)
    features = [list(arr) for arr in trainingSamples]
    
    # Convert labels to a string
    labels_str = ' '.join(map(str, labels))

    # problem = svm_problem(labels, features)

    parameters = svm_parameter()
    parameters.kernel_type = 2
    parameters.C = 1.0
    # Set other parameters as needed

    model = svm_train(labels_str, features, parameters)

    return model