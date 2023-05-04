from SupportVectorMachine import SVMOwnDataSet, SVMAgainstOtherDatasets
from NeuralNetwork import NNOwnDataSet, NNAgainstOtherDatasets

def main():
    locations = ["Kontor", "Stue", "Køkken", "Intet rum"]
    locationsFull = ["Kontor", "Stue", "Køkken", "Intet rum", "Entré" ]
    
    activationFunction = "sigmoid"
    
    fileFirstMorning = "Data/WifiData230418_9-12.txt"
    fileSecondMorning = "Data/WifiData230420_9-12.txt"
    fileThirdMorning = "Data/WifiData230421_9-12.txt"
    fileFourthMorning = "Data/WifiData230424_9-12.txt"
    fileFirstEvening = "Data/WifiData230421_17-21.txt"
    fileSecondEvening = "Data/WifiData230423_17-21.txt"
    fileThirdEvening = "Data/WifiData230424_17-21.txt"
    fileNotARoom = "Data/WifiData230413-Uni.txt"
    
    fileNameTests = [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening]
    
    # partOfData = [1, 2/3, 1/3, 2/9, 1/9]
    # minutes = [45, 30, 15, 10, 5]
    partOfData = [1]
    minutes = [45]
    
    """ NN three rooms with bias """
    
    overallPredictions = []

    for i in range(len(partOfData)): 
        predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], True, False, activationFunction)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    printMethod(overallPredictions, "NN3-B")
    
    """ NN three rooms without bias """
    
    overallPredictions = []

    for i in range(len(partOfData)):
        predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], False, False, activationFunction)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
    printMethod(overallPredictions, "NN3-UB")
    
    """ NN four rooms without bias """
    
    overallPredictions = []

    for i in range(len(partOfData)):
        predictions = predictionsNN(fileNameTests, fileNotARoom, locationsFull, partOfData[i], False, False, activationFunction)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    printMethod(overallPredictions, "NN4-UB")

def predictionsNN(fileNameTests, fileNotARoom, locations, partOfData, bias, predictedFourthRoom, activationFunction):
    predictions = []
    
    for fileName in fileNameTests:
        predict, _ = NNAgainstOtherDatasets(locations, fileName, [fileNotARoom], partOfData, bias, predictedFourthRoom, activationFunction, True)
        predictions.append(predict)
        
    return predictions
    

def printMethod(predictions, modelType):
    print(f"****************** {modelType} ******************")
    print()
    print(predictions)
    print()
    
if __name__ == '__main__':
    main()