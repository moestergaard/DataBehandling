import numpy as np
from ExtractData import getSamplesAndLabelsFromOneFile, getSamplesAndLabelsFromMultipleFiles, extractData, extractDistinctBSSIDAndNumberOfDataPoints, extractDataFromMultipleFiles
from MatrixManipulation import deterministicSplitMatrix, randomSplitSamplesAndLabels

def NNOwnDataSet(locations, filename, partOfData, bias, predictsFourthRoom, activationFunction):
    
    trainingSamplesOverall, testSamplesOverall, trainingLabelsOverall, testLabelsOverall = getSamplesAndLabelsFromOneFile(locations, filename, partOfData)
    
    wh, bh, wo, bo, percentageSure, accuracy = bestModelNN(trainingSamplesOverall, trainingLabelsOverall, bias, activationFunction, numberOfClasses=len(locations))
    
    fiveFractile = np.percentile(percentageSure, 5)
        
    if (predictsFourthRoom):
        accuracyFourthRoom, numberOfTestPointsFourthRoom = getAccuracyFourthRoom(locations, filename, partOfData, wh, bh, wo, bo, activationFunction, fiveFractile)    
    
    if partOfData == 1: 
        if (predictsFourthRoom):
            numberOfTestPoints = int(np.floor(trainingLabelsOverall.shape[0] * 0.2))
            numberOfTestPointsTotal = numberOfTestPoints + numberOfTestPointsFourthRoom
            accuracy = accuracy * numberOfTestPoints / (numberOfTestPointsTotal) + accuracyFourthRoom * numberOfTestPointsFourthRoom / numberOfTestPointsTotal
        
        return accuracy, fiveFractile
    
    predictedLabels, _ = getPredictedLabelsNN(testSamplesOverall, wh, bh, wo, bo, activationFunction, fiveFractile)
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    if (predictsFourthRoom):
        numberOfTestPoints = (trainingLabelsOverall.shape[0] - int(np.ceil(trainingLabelsOverall.shape[0] * partOfData))) * 0.2
        numberOfTestPointsTotal = numberOfTestPoints + numberOfTestPointsFourthRoom
        accuracy = accuracy * numberOfTestPoints / (numberOfTestPointsTotal) + accuracyFourthRoom * numberOfTestPointsFourthRoom / numberOfTestPointsTotal
    
    # fiveFractile = np.percentile(percentageSure, 5)*100
    
    return accuracy, fiveFractile
    

def NNAgainstOtherDatasets(locations, filename, filenameTests, partOfData, bias, predictsFourthRoom, activationFunction, testNotARoom = False):

    trainingSamples, testSamplesOverall, trainingLabels, testLabelsOverall = getSamplesAndLabelsFromMultipleFiles(locations, filename, filenameTests, partOfData, testNotARoom)
    
    wh, bh, wo, bo, percentageSure, _ = bestModelNN(trainingSamples, trainingLabels, bias, activationFunction, numberOfClasses=len(locations))
    
    fiveFractile = np.percentile(percentageSure, 5)
    fiveFractile = 0.5
    
    predictedLabels, _ = getPredictedLabelsNN(testSamplesOverall, wh, bh, wo, bo, activationFunction, fiveFractile, predictsFourthRoom)
    
    accuracy = testingNN(testLabelsOverall, predictedLabels)
    
    if (predictsFourthRoom):
        accuracyFourthRoom, numberOfTestPointsFourthRoom = getAccuracyFourthRoomTestFile(locations, filename, filenameTests, partOfData, wh, bh, wo, bo, activationFunction, fiveFractile)    
        if (partOfData == 1):
            numberOfTestPoints = int(np.floor(testLabelsOverall.shape[0] * 0.2))
        else:
            numberOfTestPoints = (testLabelsOverall.shape[0] - int(np.ceil(testLabelsOverall.shape[0] * partOfData))) * 0.2
        numberOfTestPointsTotal = numberOfTestPoints + numberOfTestPointsFourthRoom
        accuracy = accuracy * numberOfTestPoints / (numberOfTestPointsTotal) + accuracyFourthRoom * numberOfTestPointsFourthRoom / numberOfTestPointsTotal
    
    return accuracy, fiveFractile


def getAccuracyFourthRoom(locations, filename, partOfData, wh, bh, wo, bo, activationFunction, fiveFractile):
    distinctBSSID, _ = extractDistinctBSSIDAndNumberOfDataPoints(locations, filename)
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(["___", "___", "___", "___", "Entré"], filename, distinctBSSID)
    samples, labels = extractData(["___", "___", "___", "___", "Entré"], filename, distinctBSSID, dataPoints)

    trainingSamples, _, trainingLabels, _ = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    predictedLabels, _ = getPredictedLabelsNN(trainingSamples, wh, bh, wo, bo, activationFunction, fiveFractile)
    accuracyFourthRoom = testingNN(trainingLabels, predictedLabels)
    
    # if partOfData == 1:
    #     numberOfTestPoints = int(np.floor(trainingLabelsOverall.shape[0] * 0.2))
    # else: 
    #     numberOfTestPoints = (trainingLabelsOverall.shape[0] - int(np.ceil(trainingLabelsOverall.shape[0] * partOfData))) * 0.2
        
    numberOfTestPointsFourthRoom = int(np.floor(trainingLabels.shape[0] * 0.2))
    
    # numberOfTestPointsTotal = numberOfTestPoints + numberOfTestPointsFourthRoom
    
    # accuracy = accuracy * numberOfTestPoints / (numberOfTestPointsTotal) + accuracyFourthRoom * numberOfTestPointsFourthRoom / numberOfTestPointsTotal
    
    return accuracyFourthRoom, numberOfTestPointsFourthRoom

def getAccuracyFourthRoomTestFile(locations, filename, filenameTests, partOfData, wh, bh, wo, bo, activationFunction, fiveFractile):
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, filename)
    trainingSamples, trainingLabels = extractData(locations, filename, distinctBSSID, dataPoints)
    
    testSamplesOverall, testLabelsOverall = extractDataFromMultipleFiles(["___", "___", "___",  "___", "Entré"], filenameTests, distinctBSSID)
    
    # distinctBSSID, _ = extractDistinctBSSIDAndNumberOfDataPoints(locations, filename)
    # distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(["___", "___", "___", "Entré"], filename, distinctBSSID)
    # samples, labels = extractData(["___", "___", "___", "Entré"], filename, distinctBSSID, dataPoints)

    trainingSamples, _, trainingLabels, _ = randomSplitSamplesAndLabels(testSamplesOverall, testLabelsOverall, partOfData)
    
    predictedLabels, _ = getPredictedLabelsNN(trainingSamples, wh, bh, wo, bo, activationFunction, fiveFractile)
    accuracyFourthRoom = testingNN(trainingLabels, predictedLabels)
    
    # if partOfData == 1:
    #     numberOfTestPoints = int(np.floor(trainingLabelsOverall.shape[0] * 0.2))
    # else: 
    #     numberOfTestPoints = (trainingLabelsOverall.shape[0] - int(np.ceil(trainingLabelsOverall.shape[0] * partOfData))) * 0.2
        
    numberOfTestPointsFourthRoom = int(np.floor(trainingLabels.shape[0] * 0.2))
    
    # numberOfTestPointsTotal = numberOfTestPoints + numberOfTestPointsFourthRoom
    
    # accuracy = accuracy * numberOfTestPoints / (numberOfTestPointsTotal) + accuracyFourthRoom * numberOfTestPointsFourthRoom / numberOfTestPointsTotal
    
    return accuracyFourthRoom, numberOfTestPointsFourthRoom

def bestModelNN(samples, labels, bias, activationFunction, numberOfClasses):
    bestAccuracy = float('-inf')
    bestwh = None
    bestbh = None
    bestwo = None
    bestbo = None
    bestPercentageSure = None
    
    for i in range(1,6):
        trainingSamples, testSamples, trainingLabels, testLabels = deterministicSplitMatrix(samples, labels, 1/5, i)
        wh, bh, wo, bo = trainingModelNN(trainingSamples, trainingLabels, bias, activationFunction, numberOfClasses)    
        
        predictedLabels, percentageSure = getPredictedLabelsNN(testSamples, wh, bh, wo, bo, activationFunction)
        accuracy = testingNN(testLabels, predictedLabels)
        
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestwh = wh
            bestbh = bh
            bestwo = wo
            bestbo = bo
            bestPercentageSure = percentageSure
    
    return bestwh, bestbh, bestwo, bestbo, bestPercentageSure, bestAccuracy


def trainingModelNN(trainingSamples, labelsTrainingSamples, bias, activationFunction, numberOfClasses):
    # one_hot_labels = np.eye(numberOfClasses)[labelsTrainingSamples]
    one_hot_labels = np.zeros((len(labelsTrainingSamples), numberOfClasses))

    attributes = trainingSamples.shape[1]
    hidden_nodes = 4
    output_labels = numberOfClasses

    np.random.seed(42)

    wh = np.random.randn(attributes, hidden_nodes)
    if bias:
        bh = np.random.randn(hidden_nodes)
    else:
        bh = np.zeros(hidden_nodes)

    wo = np.random.randn(hidden_nodes, output_labels)
    if bias:
        bo = np.random.randn(output_labels)
    else:
        bo = np.zeros(output_labels)

    lr = 10e-4

    error_cost = []
    wh_list = []
    wo_list = []
    bh_list = []
    bo_list = []

    for epoch in range(5000):
        # feedforward
        zh = np.dot(trainingSamples, wh) + bh
        if (activationFunction == 'sigmoid'):
            ah = sigmoid(zh)
        else:
            ah = zh  # identity activation function
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)
        # ao = zo

        # # cross-entropy loss
        # loss = -np.sum(one_hot_labels * (ao - np.max(ao, axis=1, keepdims=True)))
        # loss += np.sum(np.log(np.sum(np.exp(ao - np.max(ao, axis=1, keepdims=True)), axis=1)))
        # loss /= len(labelsTrainingSamples)
        # error_cost.append(loss)

########## Phase 1

#         dcost_dzo = ao - one_hot_labels
#         dzo_dwo = ah

#         dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

#         if (bias):
#             dcost_bo = dcost_dzo

#     ########## Phases 2

#         dzo_dah = wo
#         dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
#         dah_dzh = sigmoid_der(zh)
#         dzh_dwh = trainingSamples
#         dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        # backpropagation
        dcost_dzo = ao - one_hot_labels
        dzo_dwo = ah
        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        if (bias):
            dcost_bo = dcost_dzo
            
        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dzh_dwh = trainingSamples
        
        if (activationFunction == 'sigmoid'):
            dah_dzh = sigmoid_der(zh)
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        
        else:
            # dcost_dah *= 1.0  # identity activation function
            dah_dzh = 1.0  # identity activation function
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        
        if (bias):
            dcost_bh = dcost_dah * dah_dzh

        # # backpropagation
        # dcost_dzo = ao - one_hot_labels
        # dzo_dwo = ah
        
        # dzo_dao = np.ones_like(ao)
        # dzo_dao /= np.sum(np.exp(ao - np.max(ao, axis=1, keepdims=True)), axis=1, keepdims=True)
        # dzo_dao *= np.exp(ao - np.max(ao, axis=1, keepdims=True))
        # dcost_dao = -(one_hot_labels - dzo_dao)

        # dzo_dwo = ah
        # dcost_wo = np.dot(dzo_dwo.T, dcost_dao)

        # dzo_dah = wo
        # dcost_dah = np.dot(dcost_dao, dzo_dah.T)
        # dzh_dwh = trainingSamples

        # if (activationFunction == 'sigmoid'):
        #     dah_dzh = sigmoid_der(zh)
        #     dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        
        # else:
        #     dcost_dah *= 1.0  # identity activation function
        #     dcost_wh = np.dot(dzh_dwh.T, dcost_dah)

        # if bias:
        #     dcost_bo = dcost_dao.sum(axis=0)

        # update weights
        wh -= lr * dcost_wh
        wo -= lr * dcost_wo
        if bias:
            # bh -= lr * dcost_dah.sum(axis=0)
            bh -= lr * dcost_bh.sum(axis=0)
            bo -= lr * dcost_bo.sum(axis=0)

        # store weights
        if epoch % 200 == 0:
            loss = -np.sum(one_hot_labels * np.log(ao))
            error_cost.append(loss)
#             wh_list.append(wh.copy())
#             wo_list.append(wo.copy())
#             bh_list.append(bh.copy())
#             bo_list.append(bo.copy())
        
            wh_list.append(wh.copy())
            wo_list.append(wo.copy())
            bh_list.append(bh.copy())
            bo_list.append(bo.copy())

    # select the best weights based on the lowest error cost
    i = np.argmin(error_cost)

    return wh_list[i], bh_list[i], wo_list[i], bo_list[i]


    
# def trainingModelSigmoidNN(trainingSamples, labelsTrainingSamples, bias, numberOfClasses):
#     one_hot_labels = np.zeros((len(labelsTrainingSamples), numberOfClasses))

#     for i in range(len(labelsTrainingSamples)):
#         one_hot_labels[i, labelsTrainingSamples[i].astype(int)] = 1

#     attributes = trainingSamples.shape[1]
#     hidden_nodes = 4
#     output_labels = numberOfClasses

#     np.random.seed(42)

#     wh = np.random.randn(attributes,hidden_nodes)
#     if (bias):
#         bh = np.random.randn(hidden_nodes)
#     else: bh = np.zeros(hidden_nodes)

#     wo = np.random.randn(hidden_nodes,output_labels)
#     if (bias):
#         bo = np.random.randn(output_labels)
#     else: bo = np.zeros(output_labels)
    
#     lr = 10e-4

#     error_cost = []
#     wh_list = []   
#     wo_list = []
#     bh_list = []
#     bo_list = []

#     for epoch in range(5000):
#     ############# feedforward

#         # Phase 1
#         zh = np.dot(trainingSamples, wh) + bh
#         ah = sigmoid(zh)
#         # ah = zh

#         # Phase 2
#         zo = np.dot(ah, wo) + bo
#         ao = softmax(zo)
#         # ao = zo

#     ########## Back Propagation

#     ########## Phase 1

#         dcost_dzo = ao - one_hot_labels
#         dzo_dwo = ah

#         dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

#         if (bias):
#             dcost_bo = dcost_dzo

#     ########## Phases 2

#         dzo_dah = wo
#         dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
#         dah_dzh = sigmoid_der(zh)
#         dzh_dwh = trainingSamples
#         dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

#         if (bias):
#             dcost_bh = dcost_dah * dah_dzh

#         # Update Weights ================

#         wh -= lr * dcost_wh
#         if (bias):
#             bh -= lr * dcost_bh.sum(axis=0)

#         wo -= lr * dcost_wo
#         if (bias):
#             bo -= lr * dcost_bo.sum(axis=0)
        
#         if epoch % 200 == 0:
#             loss = -np.sum(one_hot_labels * np.log(ao))
#             error_cost.append(loss)
#             wh_list.append(wh.copy())
#             wo_list.append(wo.copy())
#             bh_list.append(bh.copy())
#             bo_list.append(bo.copy())
        
#     i = np.argmin(error_cost)
    
#     return wh_list[i], bh_list[i], wo_list[i], bo_list[i]

def getPredictedLabelsNN(testSamples, wh, bh, wo, bo, activationFunction, fiveFractile = 0, predictsFourthRoom = False):
    predictedLabels = []
    percentageSure = []

    zh = np.dot(testSamples, wh) + bh
    if (activationFunction == "sigmoid"):
        ah = sigmoid(zh)
    else:
        ah = zh

    z0 = np.dot(ah, wo) + bo
    ah = softmax(z0)

    for i in range(len(ah)):
        maxPercentage = np.max(ah[i])
        if (maxPercentage < fiveFractile):
            if (predictsFourthRoom):
                predictedLabels.append(4)
            else: predictedLabels.append(3)
        else:
            predictedLabels.append(np.argmax(ah[i]))
            percentageSure.append(maxPercentage)

    return predictedLabels, percentageSure
    
    
def softmax(A):
    A -= np.max(A, axis=1, keepdims=True)
    expA = np.exp(A)
    return expA / np.sum(expA, axis=1, keepdims=True)
    # expA = np.exp(A)
    # return expA / expA.sum(axis=1, keepdims=True)


def sigmoid(x):
    # for i in range (x.shape[0]):
    #     for j in range (x.shape[1]):
    #         if x[i,j] >= 0:
    #             x[i,j] = 1 / (1 + np.exp(-x[i,j]))
    #         else:
    #             x[i,j] = np.exp(x[i,j]) / (1 + np.exp(x[i,j]))
    
    # return x
    x = 1/(1+np.exp(-x))
    return x
    


def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))


def testingNN(labelsTestSamples, predictedLabels):
    correct = 0
    
    for i in range(0, len(labelsTestSamples)):
        if labelsTestSamples[i] == predictedLabels[i]:
            correct += 1
    
    accuracy = correct/len(labelsTestSamples)
    
    return accuracy        
    