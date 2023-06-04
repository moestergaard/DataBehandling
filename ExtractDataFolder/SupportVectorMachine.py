from sklearn import svm
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import shuffleMatrices
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



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
    
    samples, labels = shuffleMatrices(samples, labels)
    
    # Split the data into training and testing sets
    trainingSamples, testSamples, trainingLabels, testLabels = train_test_split(samples, labels, test_size=0.2, random_state=42)

    # Set up the parameter grid for tuning
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.000001, 0.000015, 0.00001, 0.0001, 0.001, 0.01]}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm.SVC(cache_size = 1000, class_weight='balanced', decision_function_shape='ovo'), param_grid, cv=5)
    grid_search.fit(trainingSamples, trainingLabels)

    # Get the best trained model
    bestModel = grid_search.best_estimator_

    # Evaluate the best model on the testing set
    predictedLabels = bestModel.predict(testSamples)
    accuracy = accuracy_score(testLabels, predictedLabels)
    
    return bestModel, accuracy
        
        
def fitModel(trainingSamples, labelsTrainingSamples):
    clf = svm.SVC(cache_size = 1000, class_weight='balanced', decision_function_shape='ovo')
    clf.fit(trainingSamples, labelsTrainingSamples)

    return clf


