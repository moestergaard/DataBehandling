import joblib
import pickle
from ExtractData import getSamplesAndLabelsFromOneFile, extractDistinctBSSIDAndNumberOfDataPoints, extractData, randomSplitSamplesAndLabels, getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix
from SupportVectorMachine import bestModelSVM, fitModel
from NeuralNetwork import bestModelNN
import libsvm.svmutil as svmutil
from sklearn.datasets import dump_svmlight_file
# import svmutil

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    dataSet = "Data/WifiData230424_17-21.txt"
    # dataSetTest = "Data/WifiData230424_9-12.txt"
    
    partOfData = 1/9 # 5 minutes
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, dataSet)
    samples, labels = extractData(locations, dataSet, distinctBSSID, dataPoints)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    # trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, dataSet, partOfData)

    # bestModel, _ = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
    bestScore = float('-inf')
    bestSamples = None
    bestLabels = None
    bestModel = None
    
    for i in range(1,6):
        trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(trainingSamplesOverall, trainingLabelsOverall, 1/5, i)
        model = fitModel(trainingSamples, trainingLabels)
        
        score = model.score(testSamples, testLabels)
        
        if score > bestScore:
            bestScore = score
            bestSamples = trainingSamples
            bestLabels = trainingLabels
            bestModel = model
    
    # return bestModel, bestScore

    score = bestModel.score(testSamplesOverall, testLabelsOverall)
    
    print(score)
    print(distinctBSSID)
    print()
    
    samples = ""
    for i in range(len(bestSamples)):
        samples += "["
        for j in range(len(bestSamples[i])):
            samples += str(bestSamples[i][j]) + ", "
        samples += "] "
        # samples += str(bestSamples[i]) + " "
        # print(bestSamples[i])
    
    print(samples)
    print()
        
    labels = ""
    for i in range(len(bestLabels)):
        labels += str(bestLabels[i]) + ", "
    
    print(labels)
    
    
    # GET THE NN MODEL
    wh, bh, wo, bo, percentageSure, _ = bestModelNN(bestSamples, bestLabels, bias = False, activationFunction = 'identity', numberOfClasses=len(locations))
    
    result = printMatrice(wh)
    print(result)
    print()
    result = printMatrice(wo)
    print(result)
    print()
    print(len(bh))
    
    
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(testSamplesOverall, testLabelsOverall, 1/18)
    
    samples = ""
    for i in range(len(trainingSamplesOverall)):
        samples += "["
        for j in range(len(trainingSamplesOverall[i])):
            samples += str(trainingSamplesOverall[i][j]) + ", "
        samples += "] "
        # samples += str(bestSamples[i]) + " "
        # print(bestSamples[i])
    
    print(samples)
    print()
        
    labels = ""
    for i in range(len(trainingLabelsOverall)):
        labels += str(trainingLabelsOverall[i]) + ", "
    
    print(labels)
    
    

def printMatrice(matrice):
    return ''.join([f'{{{", ".join([f"{col:.15f}" for col in row])}}}' for row in matrice])

    
# def printMatrice(matrice):
#     for row in matrice:
#         print('{', end='')
#         for col in row:
#             print(f'{col:.15f}, ', end='')
#         print('}')

    
    
    
    
    # print(bestSamples)
    # print()
    # print(bestLabels)
    
    # dump_svmlight_file(bestSamples, bestLabels, 'svm_model.libsvm')
    
    # Export the model to a pickle file
    # with open('svm_model.libsvm', 'w') as f:
        # svmutil.svm_save_model(f.name.encode(), bestModel)
    #    dump_svmlight_file(bestModel, labels, 'svm_model.libsvm')
        # pickle.dump(bestModel, f)


if __name__ == '__main__':
    main()