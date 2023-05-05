from SupportVectorMachine import SVMOwnDataSet, SVMAgainstOtherDatasets
from NeuralNetwork import NNOwnDataSet, NNAgainstOtherDatasets

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    locationsFull = ["Kontor", "Stue", "Køkken", "Entré"]
    
    activationFunction = ["identity"]
    
    fileFirstMorning = "Data/WifiData230418_9-12.txt"
    fileSecondMorning = "Data/WifiData230420_9-12.txt"
    fileThirdMorning = "Data/WifiData230421_9-12.txt"
    fileFourthMorning = "Data/WifiData230424_9-12.txt"
    fileFirstEvening = "Data/WifiData230421_17-21.txt"
    fileSecondEvening = "Data/WifiData230423_17-21.txt"
    fileThirdEvening = "Data/WifiData230424_17-21.txt"
    
    partOfData = [1, 2/3, 1/3, 2/9, 1/9]
    minutes = [45, 30, 15, 10, 5]
    # partOfData = [1]
    # minutes = [45]
    
    
    # """ SVM three rooms """
    
    # overallPredictions = []
    
    # for i in range(len(partOfData)):
    #     ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, predictions = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i])
    #     predictionsUpdated = changePredictions(predictions)
    #     overallPredictions.append(predictionsUpdated)
    #     #printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM3")
    
    # printMethod(overallPredictions, "SVM3")
    # # print(f"****************** SVM3 ******************")
    # # print()
    # # print(overallPredictions)
    # # # print(overallPredictions[0])
    # # # for predictions in overallPredictions:
    # # #     print(predictions[0])
    # # print()
    
    # """ SVM four rooms """
    
    # overallPredictions = []
    
    # for i in range(len(partOfData)):
    #     ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, predictions = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i])
    #     predictionsUpdated = changePredictions(predictions)
    #     overallPredictions.append(predictionsUpdated)
    #     # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM4")
    
    # printMethod(overallPredictions, "SVM4")
        
    for j in range(len(activationFunction)):
        print(f"****************** {activationFunction[j]} ******************")
        """ NN three rooms with bias """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True, False, activationFunction[j])
            predictionsUpdated = changePredictions(predictions)
            overallPredictions.append(predictionsUpdated)
            # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
        printMethod(overallPredictions, "NN3-B")
        
        """ NN three rooms with bias - predicts fourth room """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True, True, activationFunction[j])
            predictionsUpdated = changePredictions(predictions)
            overallPredictions.append(predictionsUpdated)
            # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B-Predicts", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
        printMethod(overallPredictions, "NN3-B-P")
        
        """ NN three rooms without bias """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False, False, activationFunction[j])
            predictionsUpdated = changePredictions(predictions)
            overallPredictions.append(predictionsUpdated)
            # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
            
        printMethod(overallPredictions, "NN3-UB")
        
        """ NN three rooms without bias - predicts fourth room """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False, True, activationFunction[j])
            predictionsUpdated = changePredictions(predictions)
            overallPredictions.append(predictionsUpdated)
            # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB-Predicts", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
            
        printMethod(overallPredictions, "NN3-UB-P")
        
        """ NN four rooms without bias """
        
        overallPredictions = []

        for i in range(len(partOfData)):
            ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i], False, False, activationFunction[j])
            predictionsUpdated = changePredictions(predictions)
            overallPredictions.append(predictionsUpdated)
            # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
        printMethod(overallPredictions, "NN4-UB")
    
def predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData):
    predictions = []
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = 0
    
    """ Morning own dataset """
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, partOfData)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, partOfData)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, partOfData)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, partOfData)

    # Append all ownDsM to predictions
    predictions.append([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    
    # print([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1 = SVMOwnDataSet(locations, fileFirstEvening, partOfData)
    ownDsE2 = SVMOwnDataSet(locations, fileSecondEvening, partOfData)
    ownDsE3 = SVMOwnDataSet(locations, fileThirdEvening, partOfData)

    predictions.append([ownDsE1, ownDsE2, ownDsE3])
    # print([ownDsE1, ownDsE2, ownDsE3])
    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    
    """ Morning other dataset """
    
    otherDayM11 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning], partOfData)
    otherDayM12 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdMorning], partOfData)
    otherDayM13 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFourthMorning], partOfData)
    otherDayM21 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning], partOfData)
    otherDayM22 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdMorning], partOfData)
    otherDayM23 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFourthMorning], partOfData)
    otherDayM31 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning], partOfData)
    otherDayM32 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileSecondMorning], partOfData)
    otherDayM33 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFourthMorning], partOfData)
    otherDayM41 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning], partOfData)
    otherDayM42 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondMorning], partOfData)
    otherDayM43 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdMorning], partOfData)
    
    
    predictions.append([otherDayM11, otherDayM12, otherDayM13, otherDayM21, otherDayM22, otherDayM23, otherDayM31, otherDayM32, otherDayM33, otherDayM41, otherDayM42, otherDayM43])
    otherDayM = (otherDayM11 + otherDayM12 + otherDayM13 + otherDayM21 + otherDayM22 + otherDayM23 + otherDayM31 + otherDayM32 + otherDayM33 + otherDayM41 + otherDayM42 + otherDayM43) / 12
    
    """ Morning other time """
    
    otherTimeM11 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening], partOfData)
    otherTimeM12 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondEvening], partOfData)
    otherTimeM13 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdEvening], partOfData)
    otherTimeM21 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening], partOfData)
    otherTimeM22 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileSecondEvening], partOfData)
    otherTimeM23 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdEvening], partOfData)
    otherTimeM31 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData)
    otherTimeM32 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileSecondEvening], partOfData)
    otherTimeM33 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileThirdEvening], partOfData)
    otherTimeM41 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening], partOfData)
    otherTimeM42 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondEvening], partOfData)
    otherTimeM43 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdEvening], partOfData)
    
    predictions.append([otherTimeM11, otherTimeM12, otherTimeM13, otherTimeM21, otherTimeM22, otherTimeM23, otherTimeM31, otherTimeM32, otherTimeM33, otherTimeM41, otherTimeM42, otherTimeM43])
    otherTimeM = (otherTimeM11 + otherTimeM12 + otherTimeM13 + otherTimeM21 + otherTimeM22 + otherTimeM23 + otherTimeM31 + otherTimeM32 + otherTimeM33 + otherTimeM41 + otherTimeM42 + otherTimeM43) / 12
    
    """ Evening other dataset """
    
    otherDayE11 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening], partOfData)
    otherDayE12 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdEvening], partOfData)
    otherDayE21 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening], partOfData)
    otherDayE22 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdEvening], partOfData)
    otherDayE31 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening], partOfData)
    otherDayE32 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondEvening], partOfData)
    
    predictions.append([otherDayE11, otherDayE12, otherDayE21, otherDayE22, otherDayE31, otherDayE32])
    otherDayE = (otherDayE11 + otherDayE12 + otherDayE21 + otherDayE22 + otherDayE31 + otherDayE32) / 6
    
    
    """ Evening other time """
    
    otherTimeE11 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning], partOfData)
    otherTimeE12 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondMorning], partOfData)
    otherTimeE13 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdMorning], partOfData)
    otherTimeE14 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFourthMorning], partOfData)
    otherTimeE21 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning], partOfData)
    otherTimeE22 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileSecondMorning], partOfData)
    otherTimeE23 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdMorning], partOfData)
    otherTimeE24 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFourthMorning], partOfData)
    otherTimeE31 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning], partOfData)
    otherTimeE32 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondMorning], partOfData)
    otherTimeE33 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileThirdMorning], partOfData)
    otherTimeE34 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFourthMorning], partOfData)
    
    predictions.append([otherTimeE11, otherTimeE12, otherTimeE13, otherTimeE14, otherTimeE21, otherTimeE22, otherTimeE23, otherTimeE24, otherTimeE31, otherTimeE32, otherTimeE33, otherTimeE34])
    otherTimeE = (otherTimeE11 + otherTimeE12 + otherTimeE13 + otherTimeE14 + otherTimeE21 + otherTimeE22 + otherTimeE23 + otherTimeE24 + otherTimeE31 + otherTimeE32 + otherTimeE33 + otherTimeE34) / 12
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, predictions

def predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData, bias, predictedFourthRoom, activationFunction):
    predictions = []
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = ownFiveFractileM = ownFiveFractileE = 0
    
    """ Morning own dataset """
    
    ownDsM1, ownFiveFractileM1 = NNOwnDataSet(locations, fileFirstMorning, partOfData, bias, predictedFourthRoom, activationFunction)
    ownDsM2, ownFiveFractileM2 = NNOwnDataSet(locations, fileSecondMorning, partOfData, bias, predictedFourthRoom, activationFunction)
    ownDsM3, ownFiveFractileM3 = NNOwnDataSet(locations, fileThirdMorning, partOfData, bias, predictedFourthRoom, activationFunction)
    ownDsM4, ownFiveFractileM4 = NNOwnDataSet(locations, fileFourthMorning, partOfData, bias, predictedFourthRoom, activationFunction)

    predictions.append([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    ownFiveFractileM = (ownFiveFractileM1 + ownFiveFractileM2 + ownFiveFractileM3 + ownFiveFractileM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1, ownFiveFractileE1 = NNOwnDataSet(locations, fileFirstEvening, partOfData, bias, predictedFourthRoom, activationFunction)
    ownDsE2, ownFiveFractileE2 = NNOwnDataSet(locations, fileSecondEvening, partOfData, bias, predictedFourthRoom, activationFunction)
    ownDsE3, ownFiveFractileE3 = NNOwnDataSet(locations, fileThirdEvening, partOfData, bias, predictedFourthRoom, activationFunction)

    predictions.append([ownDsE1, ownDsE2, ownDsE3])
    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    ownFiveFractileE = (ownFiveFractileE1 + ownFiveFractileE2 + ownFiveFractileE3) / 3
    
    """ Morning other dataset """
    
    otherDayM11, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM12, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM13, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM21, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM22, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM23, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM31, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM32, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM33, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM41, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM42, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayM43, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    
    predictions.append([otherDayM11, otherDayM12, otherDayM13, otherDayM21, otherDayM22, otherDayM23, otherDayM31, otherDayM32, otherDayM33, otherDayM41, otherDayM42, otherDayM43])
    otherDayM = (otherDayM11 + otherDayM12 + otherDayM13 + otherDayM21 + otherDayM22 + otherDayM23 + otherDayM31 + otherDayM32 + otherDayM33 + otherDayM41 + otherDayM42 + otherDayM43) / 12
    otherDayFiveFractileM = (otherDayFiveFractileM1 + otherDayFiveFractileM2 + otherDayFiveFractileM3 + otherDayFiveFractileM4) / 4
    
    """ Morning other time """
    
    otherTimeM11, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM12, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM13, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM21, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM22, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM23, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM31, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM32, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM33, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM41, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM42, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeM43, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    
    predictions.append([otherTimeM11, otherTimeM12, otherTimeM13, otherTimeM21, otherTimeM22, otherTimeM23, otherTimeM31, otherTimeM32, otherTimeM33, otherTimeM41, otherTimeM42, otherTimeM43])
    otherTimeM = (otherTimeM11 + otherTimeM12 + otherTimeM13 + otherTimeM21 + otherTimeM22 + otherTimeM23 + otherTimeM31 + otherTimeM32 + otherTimeM33 + otherTimeM41 + otherTimeM42 + otherTimeM43) / 12
    otherTimeFiveFractileM = (otherTimeFiveFractileM1 + otherTimeFiveFractileM2 + otherTimeFiveFractileM3 + otherTimeFiveFractileM4) / 4
    
    """ Evening other dataset """
    
    otherDayE11, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayE12, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayE21, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayE22, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayE31, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    otherDayE32, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom, activationFunction)
    
    predictions.append([otherDayE11, otherDayE12, otherDayE21, otherDayE22, otherDayE31, otherDayE32])
    otherDayE = (otherDayE11 + otherDayE12 + otherDayE21 + otherDayE22 + otherDayE31 + otherDayE32) / 6
    otherDayFiveFractileE = (otherDayFiveFractileE1 + otherDayFiveFractileE2 + otherDayFiveFractileE3) / 3
    
    """ Evening other time """
    
    otherTimeE11, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE12, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE13, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE14, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE21, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE22, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE23, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE24, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE31, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE32, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE33, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    otherTimeE34, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom, activationFunction)
    
    predictions.append([otherTimeE11, otherTimeE12, otherTimeE13, otherTimeE14, otherTimeE21, otherTimeE22, otherTimeE23, otherTimeE24, otherTimeE31, otherTimeE32, otherTimeE33, otherTimeE34])
    otherTimeE = (otherTimeE11 + otherTimeE12 + otherTimeE13 + otherTimeE14 + otherTimeE21 + otherTimeE22 + otherTimeE23 + otherTimeE24 + otherTimeE31 + otherTimeE32 + otherTimeE33 + otherTimeE34) / 12
    otherTimeFiveFractileE = (otherTimeFiveFractileE1 + otherTimeFiveFractileE2 + otherTimeFiveFractileE3) / 3
    
    return ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions
    

def changePredictions(predictions):
    predictionsUpdated = []
    for predict in predictions:
        for p in predict:
            predictionsUpdated.append(p)
    return predictionsUpdated

def printMethod(predictions, modelType):
    print(f"****************** {modelType} ******************")
    print()
    print(predictions)
    print()
    

# def printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes, modelType, ownFiveFractileM = 0, ownFiveFractileE = 0, otherDayFiveFractileM = 0, otherDayFiveFractileE = 0, otherTimeFiveFractileM = 0, otherTimeFiveFractileE = 0):
#     print(f"****************** {modelType} ******************")
#     print("Antal minutter: ", minutes)
#     print()
#     print("Nøjagtighed egen dataindsamling")
#     print("Morgen: %2.2f procent" % (ownDsM*100))
#     if (ownFiveFractileM != 0):
#         print("5-fraktil morgen: %2.2f procent" % (ownFiveFractileM*100))
#     print("Aften: %2.2f procent" % (ownDsE*100))
#     if (ownFiveFractileE != 0):
#         print("5-fraktil aften: %2.2f procent" % (ownFiveFractileE*100))
#     print()
#     print("Nøjagtighed anden dag")
#     print("Morgen: %2.2f procent" % (otherDayM*100))
#     if (otherDayFiveFractileM != 0):
#         print("5-fraktil morgen: %2.2f procent" % (otherDayFiveFractileM*100))
#     print("Aften: %2.2f procent" % (otherDayE*100))
#     if (otherDayFiveFractileE != 0):
#         print("5-fraktil aften: %2.2f procent" % (otherDayFiveFractileE*100))
#     print()
#     print("Nøjagtighed andet tidspunkt")
#     print("Morgen: %2.2f procent" % (otherTimeM*100))
#     if (otherTimeFiveFractileM != 0):
#         print("5-fraktil morgen: %2.2f procent" % (otherTimeFiveFractileM*100))
#     print("Aften: %2.2f procent" % (otherTimeE*100))
#     if (otherTimeFiveFractileE != 0):
#         print("5-fraktil aften: %2.2f procent" % (otherTimeFiveFractileE*100))
#     print()
    
if __name__ == '__main__':
    main()