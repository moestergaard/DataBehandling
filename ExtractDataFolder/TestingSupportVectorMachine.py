import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from ExtractData import extractDistinctBSSIDAndNumberOfDataPoints, extractData, extractDataCombined
from MatrixManipulation import randomSplitSamplesAndLabels, deterministicSplitMatrix
from SupportVectorMachine import calculationsSVM, accuracySVM

def datasetTestedAgainstAnotherDatasetSVM(locations, filename, filenameTest, partOfData):

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, locations, distinctBSSID)

    # print(f"Distinct BSSID in {filename}: {distinctBSSID}")
    # print('Length of distinct BSSID: ', len(distinctBSSID))
    # print(f"Distinct BSSID in {filenameTest}: {distinctBSSIDTest}")
    # print('Length of distinct BSSID test: ', len(distinctBSSIDTest))

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

    # print()
    # print("********************************************************************************************")
    # print()
    # print(f"{filename} tested against {filenameTest}")
    
    accuracy = accuracySVM(testLabels, predictionSVM, numberOfClasses=len(locations))
    return accuracy


def SVMOwnDataSet(locations, filename, partOfData):
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename, locations)
    samples, labels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    bestModel, score = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    if( partOfData == 1): return score

    score = bestModel.score(testSamplesOverall, testLabelsOverall)    
    return score

    # for i in range(5):
    #     trainingSamples, testSamples, trainingLabels, testLabel = deterministicSplitMatrix(trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall, 1/5, i)
        
    #     print(" *************** ", i, " *************** ")
        
    
    
    
    # trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations, partOfData)

    # predictionSVM = calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples)
    
    # accuracy = accuracySVM(labelsTestSamples, predictionSVM, numberOfClasses = len(locations))
    # return accuracy


# def svm_cross_validation(X, y, new_data=None):
#     # Define the parameter grid to search over
#     param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    
#     # Create an SVM model
#     svm_model = SVC()

#     # Use GridSearchCV to find the best hyperparameters
#     grid_search = GridSearchCV(svm_model, param_grid=param_grid, cv=5)
#     grid_search.fit(X, y)

#     # Print the best hyperparameters and cross-validation score
#     print("Best hyperparameters: ", grid_search.best_params_)
#     print("Cross-validation score: ", grid_search.best_score_)

#     # Make predictions on new data using the best performing model
#     if new_data is not None:
#         best_model = grid_search.best_estimator_
#         predicted_labels = best_model.predict(new_data)
#         print("Predicted labels for new data: ", predicted_labels)
        
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
    # prediction = clf.predict(testSamples)

    return clf