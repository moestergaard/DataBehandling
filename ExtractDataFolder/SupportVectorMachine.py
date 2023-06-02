from sklearn import svm
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix, shuffleMatrices, randomSplitSamplesAndLabels
from libsvm.svmutil import svm_problem, svm_parameter, svm_train
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def SVMOwnDataSet(locations, filename, partOfData):
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    
    # trainingSamples, testSamples, trainingLabels, testLabels = randomSplitSamplesAndLabels(trainingSamplesOverall, trainingLabelsOverall, 0.8)
    # bestModel = fitModel(trainingSamples, trainingLabels)
    # score = bestModel.score(testSamples, testLabels)
    
    bestModel, score = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    if( partOfData == 1): return score

    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    
    return score


def SVMAgainstOtherDatasets(locations, filename, filenameTests, partOfData):

    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData)
    
    # model = fitModel(trainingSamples, trainingLabels)
    # score = model.score(testSamplesOverall, testLabelsOverall)    
    # trainingSamples, _, trainingLabels, _ = randomSplitSamplesAndLabels(trainingSamples, trainingLabels, 0.8)
    # bestModel = fitModel(trainingSamples, trainingLabels)
    
    bestModel, scoreBestModel = bestModelSVM(trainingSamples, trainingLabels)
    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    
    return score
        
        
def bestModelSVM(samples, labels):
    bestScore = float('-inf')
    bestModel = None
    
    samples, labels = shuffleMatrices(samples, labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

    # Set up the parameter grid for tuning
    # param_grid = {'C': [10, 100, 1000, 10000], 'gamma': [0.000001, 0.000015, 0.00001]}
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.000001, 0.000015, 0.00001, 0.0001, 0.001, 0.01]}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm.SVC(cache_size = 1000, class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best trained model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the testing set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # print("Best Model: ", best_model)
    # print("Accuracy: ", accuracy)
    
    
    # for i in range(1,6):
    #     trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(samples, labels, 1/5, i)
    #     model = fitModel(trainingSamples, trainingLabels)
        
    #     score = model.score(testSamples, testLabels)
        
    #     if score > bestScore:
    #         bestScore = score
    #         bestModel = model
    
    return best_model, accuracy
        
        
def fitModel(trainingSamples, labelsTrainingSamples):
    clf = svm.SVC(cache_size = 1000, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(trainingSamples, labelsTrainingSamples)

    return clf


