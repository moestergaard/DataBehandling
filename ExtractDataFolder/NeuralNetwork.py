import numpy as np
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix

def NNOwnDataSet(locations, filename, partOfData, bias):
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    wh, bh, wo, bo, percentageSure, accuracy = bestModelNN(trainingSamplesOverall, trainingLabelsOverall, bias, numberOfClasses=len(locations))
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    if partOfData == 1: return accuracy, fiveFractile
    
    predictedLabels, percentageSure = getPredictedLabelsNN(testSamplesOverall, wh, bh, wo, bo)
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    return accuracy, fiveFractile
    

def NNAgainstOtherDatasets(locations, filename, filenameTests, partOfData, bias):

    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData)
    
    wh, bh, wo, bo, _, _ = bestModelNN(trainingSamples, trainingLabels, bias, numberOfClasses=len(locations))
    
    predictedLabels, percentageSure = getPredictedLabelsNN(testSamplesOverall, wh, bh, wo, bo)
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    fiveFractile = np.percentile(percentageSure, 5)*100
    
    return accuracy, fiveFractile


def bestModelNN(samples, labels, bias, numberOfClasses):
    bestAccuracy = float('-inf')
    bestwh = None
    bestbh = None
    bestwo = None
    bestbo = None
    bestPercentageSure = None
    
    for i in range(1,6):
        trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(samples, labels, 1/5, i)
        wh, bh, wo, bo = trainingModelNN(trainingSamples, trainingLabels, bias, numberOfClasses)
        
        predictedLabels, percentageSure = getPredictedLabelsNN(testSamples, wh, bh, wo, bo)
        accuracy = testingNN(testLabels, predictedLabels)
        
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestwh = wh
            bestbh = bh
            bestwo = wo
            bestbo = bo
            bestPercentageSure = percentageSure
    
    return bestwh, bestbh, bestwo, bestbo, bestPercentageSure, bestAccuracy
    
def trainingModelNN(trainingSamples, labelsTrainingSamples, bias, numberOfClasses):
    one_hot_labels = np.zeros((len(labelsTrainingSamples), numberOfClasses))

    for i in range(len(labelsTrainingSamples)):
        one_hot_labels[i, labelsTrainingSamples[i].astype(int)] = 1

    attributes = trainingSamples.shape[1]
    hidden_nodes = 4
    output_labels = numberOfClasses

    np.random.seed(42)

    wh = np.random.randn(attributes,hidden_nodes)
    if (bias):
        bh = np.random.randn(hidden_nodes)
    else: bh = np.zeros(hidden_nodes)

    wo = np.random.randn(hidden_nodes,output_labels)
    if (bias):
        bo = np.random.randn(output_labels)
    else: bo = np.zeros(output_labels)
    
    lr = 10e-4

    error_cost = []
    wh_list = []   
    wo_list = []
    bh_list = []
    bo_list = []

    for epoch in range(5000):
    ############# feedforward

        # Phase 1
        zh = np.dot(trainingSamples, wh) + bh
        ah = sigmoid(zh)
        # ah = zh

        # Phase 2
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)
        # ao = zo

    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = ao - one_hot_labels
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        if (bias):
            dcost_bo = dcost_dzo

    ########## Phases 2

        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = trainingSamples
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        if (bias):
            dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        wh -= lr * dcost_wh
        if (bias):
            bh -= lr * dcost_bh.sum(axis=0)

        wo -= lr * dcost_wo
        if (bias):
            bo -= lr * dcost_bo.sum(axis=0)
        
        if epoch % 200 == 0:
            loss = -np.sum(one_hot_labels * np.log(ao))
            error_cost.append(loss)
            wh_list.append(wh.copy())
            wo_list.append(wo.copy())
            bh_list.append(bh.copy())
            bo_list.append(bo.copy())
        
    i = np.argmin(error_cost)
    
    return wh_list[i], bh_list[i], wo_list[i], bo_list[i]

def getPredictedLabelsNN(testSamples, wh, bh, wo, bo):
    predictedLabels = []
    percentageSure = []

    zh = np.dot(testSamples, wh) + bh
    ah = sigmoid(zh)

    z0 = np.dot(ah, wo) + bo
    ah = softmax(z0)

    for i in range(len(ah)):
        predictedLabels.append(np.argmax(ah[i]))
        percentageSure.append(np.max(ah[i]))

    return predictedLabels, percentageSure
    
    
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))


def testingNN(labelsTestSamples, predictedLabels):
    correct = 0
    
    for i in range(0, len(labelsTestSamples)):
        if labelsTestSamples[i] == predictedLabels[i]:
            correct += 1
    
    accuracy = correct/len(labelsTestSamples)
    
    return accuracy        
    