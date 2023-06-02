import joblib
import pickle
from ExtractData import getSamplesAndLabelsFromOneFile, extractDistinctBSSIDAndNumberOfDataPoints, extractData, randomSplitSamplesAndLabels, getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix
from SupportVectorMachine import bestModelSVM, fitModel
from NeuralNetwork import bestModelNN
import libsvm.svmutil as svmutil
import libsvm
from libsvm.svmutil import svm_save_model
from sklearn.datasets import dump_svmlight_file
import json
import numpy as np
# import svmutil

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    dataSet = "Data/WifiData230424_17-21.txt"
    # dataSetTest = "Data/WifiData230424_9-12.txt"
    
    # partOfData = 1/9 # 5 minutes
    partOfData = 1/3 # 15 minutes
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, dataSet)
    samples, labels = extractData(locations, dataSet, distinctBSSID, dataPoints)
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    # trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, dataSet, partOfData)

    bestModel, _ = bestModelSVM(trainingSamplesOverall, trainingLabelsOverall)
    
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
    print("distinctBSSID: ")
    print(distinctBSSID)
    print("len(distinctBSSID)" + str(len(distinctBSSID)))
    print()
    
    # samples = ""
    # for i in range(len(bestSamples)):
    #     samples += "["
    #     for j in range(len(bestSamples[i])):
    #         samples += str(bestSamples[i][j]) + ", "
    #     samples += "] "
    #     # samples += str(bestSamples[i]) + " "
    #     # print(bestSamples[i])
    
    # print(samples)
    # print()
        
    # labels = ""
    # for i in range(len(bestLabels)):
    #     labels += str(bestLabels[i]) + ", "
    
    # print(labels)
    
    
    # GET THE NN MODEL
    wh, bh, wo, bo, percentageSure, _ = bestModelNN(bestSamples, bestLabels, bias = False, activationFunction = 'identity', numberOfClasses=len(locations))
    
    # result = printMatrice(wh)
    # print(result)
    # print()
    # result = printMatrice(wo)
    # print(result)
    # print()
    # print(len(bh))
    
    
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = randomSplitSamplesAndLabels(testSamplesOverall, testLabelsOverall, 1/18)
    
    samples = ""
    for i in range(len(trainingSamplesOverall)):
        samples += "["
        for j in range(len(trainingSamplesOverall[i])):
            samples += str(trainingSamplesOverall[i][j]) + ", "
        samples += "] "
        # samples += str(bestSamples[i]) + " "
        # print(bestSamples[i])
    
    # print(samples)
    # print()
        
    # labels = ""
    # for i in range(len(trainingLabelsOverall)):
    #     labels += str(trainingLabelsOverall[i]) + ", "
    
    # print(labels)
    
    result = bestModel.score(trainingSamplesOverall, trainingLabelsOverall)
    print("Accuracy: " + str(result))
    
    
    # storeModel(bestModel, 'svm_model1.json', bestSamples, bestLabels)
    storeModel(bestModel, 'svm_model_testLibSvm.json', bestSamples, bestLabels)
    

def storeModel2(model, filename):
    with open(filename, 'w') as f:
        json.dump(model, f)
    
def storeModel(model, filename, bestSamples, bestLabels):
    
    # svm_save_model(filename, model)
    
    # calculate rho
    # rho = model.intercept_[0]
    # for i in range(len(model.support_)):
    #     rho -= model.dual_coef_[0][i] * model._gamma(bestSamples[model.support_[i]], bestSamples[model.support_[0]])
        
    # rho = model.intercept_[0]

    # for i in range(len(model.support_)):
    #     print(model.support_[i])
    #     print(bestSamples[model.support_[i]])
    #     print(bestSamples[model.support_[0]])
    #     print(model._gamma)
    #     gamma = model._gamma(bestSamples[model.support_[i]], bestSamples[model.support_[0]])
    #     dual_coef = model.dual_coef_[0][i]
    #     rho -= dual_coef * gamma
    
    # rho = [1]
    # rho[0] = model.intercept_[0]
    # for i in range(len(model.support_)):
    #     rho[0] -= model.dual_coef_[0][i] * model._gamma
        
    # Compute the rho array
    # rho = model.decision_function(model.support_vectors_).mean()
    # print(rho)
    
    # for i in range (len(model.support_vectors_)):
    #     print(model.decision_function(model.support_vectors_))
        
    # rho = []
    # for j in range(3):
    #     temp = 0
    #     for i in range(len(model.support_vectors_)):
    #         temp += model.decision_function([model.support_vectors_[i]])[0][j]
    #         print(model.decision_function([model.support_vectors_[i]]))
    #         print(model.decision_function([model.support_vectors_[i]])[0][1])
    #     rho.append(temp/len(model.support_vectors_))
    
    # print("***")
    # print(rho)
    # print("***")
    
    # print(len(model.support_vectors_).shape())
    # print(bestSamples.shape[1])
    # rho = []
    # for i in range(bestSamples.shape[1]):
    #     rho_i = model.decision_function(model.support_vectors_[:, i]).mean()
    #     rho.append(rho_i)
    # rho = np.array(rho)

    # print(rho)
    # print(len(model.support_vectors_))
    
        
    # Extract the relevant parameters of the SVC object
    params = {
    #     "C": model.C,
        "kernel": model.kernel,
        "gamma": model._gamma
        # "coef0": model.coef0,
        # "degree": model.degree,
        # "class_weight": model.class_weight,
        # "decision_function_shape": model.decision_function_shape,
        # "tol": model.tol,
        # "max_iter": model.max_iter,
        # "random_state": model.random_state,
        # "verbose": model.verbose,
        # "cache_size": model.cache_size,
        # #"svm_type": model.svm_type,
        # "svm_nu": model.nu
    }
    
    
    # Create a dictionary containing the parameters and the support vectors and coefficients
    data = {
        "params": params,
        # "support_vectors": getSupportVectors(model),
        "coefficients": model.dual_coef_.tolist(),
        "classIndices": model.classes_.tolist(),
        # "nSV": model.n_support_.tolist(),
        # "rho": rho
        "intercepts": model.intercept_.tolist()
        
    }
    
    print()
    # print("Support vectors: ")
    # print(getSupportVectors(model).shape)
    print("Coefficients: ")
    print(model.dual_coef_.shape)
    print("Intercepts: ")
    print(model.intercept_.shape)
    print("Classes: ")
    print(model.classes_.shape)
    
    # Serialize the dictionary as JSON and store it in the specified file
    with open(filename, 'w') as f:
        json.dump(data, f)

def printMatrice(matrice):
    return ''.join([f'{{{", ".join([f"{col:.15f}" for col in row])}}}' for row in matrice])


# def getSupportVectors(model):
#     support_vectors = []
#     coef = model.coef_
#     for c in coef:
#         support_vector = -c / np.linalg.norm(c)  # Calculate the support vector
#         support_vectors.append(support_vector)
#     return support_vectors
    
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