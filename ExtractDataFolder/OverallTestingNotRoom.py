from NeuralNetwork import NNAgainstOtherDatasets

def main():
    locations = ["Kontor", "Stue", "Køkken", "Intet rum"]
    locationsFull = ["Kontor", "Stue", "Køkken", "Intet rum", "Entré" ]
    
    activationFunction = ["sigmoid", "identity"]
    
    fileFirstMorning = "Data/WifiData230418_9-12.txt"
    fileSecondMorning = "Data/WifiData230420_9-12.txt"
    fileThirdMorning = "Data/WifiData230421_9-12.txt"
    fileFourthMorning = "Data/WifiData230424_9-12.txt"
    fileFirstEvening = "Data/WifiData230421_17-21.txt"
    fileSecondEvening = "Data/WifiData230423_17-21.txt"
    fileThirdEvening = "Data/WifiData230424_17-21.txt"
    fileNotARoom = "Data/WifiData230413-Uni.txt"
    
    fileNameTests = [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening]
    
    partOfData = [1, 2/3, 1/3, 2/9, 1/9]
    
    for j in range(len(activationFunction)):
        print("Activation function: " + activationFunction[j])
        """ NN three rooms with bias """
        
        overallPredictions = []

        for i in range(len(partOfData)): 
            predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], True, False, activationFunction[j])
            overallPredictions.append(predictions)
        
        printMethod(overallPredictions, "NN3-B")
        
        """ NN three rooms with bias - predicts fourth room """
        
        overallPredictions = []

        for i in range(len(partOfData)): 
            predictions = predictionsNN(fileNameTests, fileNotARoom, locationsFull, partOfData[i], True, True, activationFunction[j])
            overallPredictions.append(predictions)
        
        printMethod(overallPredictions, "NN3-B-P")
        
        """ NN three rooms without bias """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], False, False, activationFunction[j])
            overallPredictions.append(predictions)
            
        printMethod(overallPredictions, "NN3-UB")
        
        """ NN three rooms without bias - predicts fourth room """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            predictions = predictionsNN(fileNameTests, fileNotARoom, locationsFull, partOfData[i], False, True, activationFunction[j])
            overallPredictions.append(predictions)
            
        printMethod(overallPredictions, "NN3-UB-P")
        
        """ NN four rooms without bias """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            predictions = predictionsNN(fileNameTests, fileNotARoom, locationsFull, partOfData[i], False, False, activationFunction[j])
            overallPredictions.append(predictions)
        
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