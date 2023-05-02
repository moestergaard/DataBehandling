import numpy as np

def randomSplitSamplesAndLabels(samples, labels, ratio):
    
    np.random.seed(43)
    
    m, n = samples.shape
    trainingRows = int(np.ceil(m * ratio))
    trainingIndices = np.random.choice(m, trainingRows, replace=False)
    trainingSamples = samples[trainingIndices]
    testSamples = np.delete(samples, trainingIndices, axis=0)
    trainingLabels = labels[trainingIndices]
    testLabels = np.delete(labels, trainingIndices, axis=0)
    
    return trainingSamples, testSamples, trainingLabels, testLabels


def deterministicSplitMatrix(samples, labels, ratio, splitNumber):
    m, n = samples.shape
    trainingSize = int(np.ceil(m * ratio))
    startIndex = int((splitNumber - 1) * trainingSize)
    endIndex = int(splitNumber * trainingSize)
    if endIndex > m:
        endIndex = m
    trainingSamples = samples[startIndex:endIndex]
    testSamples = np.delete(samples, np.s_[startIndex:endIndex], axis=0)
    trainingLabels = labels[startIndex:endIndex]
    testLabels = np.delete(labels, np.s_[startIndex:endIndex], axis=0)
    
    return trainingSamples, testSamples, trainingLabels, testLabels


def changeMatrice(matrice, index, distinctBSSID, currentBSSID, resultLevel):
    for i in range(0, len(distinctBSSID)):
        if distinctBSSID[i] == currentBSSID:
            matrice[index, i] = resultLevel
            return matrice  
        
    return matrice  