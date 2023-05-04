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
        predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], True, False)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    printMethod(overallPredictions, "NN3-B")
    
    """ NN three rooms without bias """
    
    overallPredictions = []

    for i in range(len(partOfData)):
        predictions = predictionsNN(fileNameTests, fileNotARoom, locations, partOfData[i], False, False)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
    printMethod(overallPredictions, "NN3-UB")
    
    """ NN four rooms without bias """
    
    overallPredictions = []

    for i in range(len(partOfData)):
        predictions = predictionsNN(fileNameTests, fileNotARoom, locationsFull, partOfData[i], False, False)
        # predictionsUpdated = changePredictions(predictions)
        overallPredictions.append(predictions)
        # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    printMethod(overallPredictions, "NN4-UB")

def predictionsNN(fileNameTests, fileNotARoom, locations, partOfData, bias, predictedFourthRoom):
    predictions = []
    
    for fileName in fileNameTests:
        predict, _ = NNAgainstOtherDatasets(locations, fileName, [fileNotARoom], partOfData, bias, predictedFourthRoom, True)
        predictions.append(predict)
        
    return predictions
    
    # m1, fiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # m2, fiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # m3, fiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # m4, fiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # e1, fiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # e2, fiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    # e3, fiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileNotARoom], partOfData, bias, predictedFourthRoom)
    
    # predictions.append([m1, m2, m3, m4, e1, e2, e3])
    
    
    
    # ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = ownFiveFractileM = ownFiveFractileE = 0
    
    # """ Morning own dataset """
    
    # ownDsM1, ownFiveFractileM1 = NNOwnDataSet(locations, fileFirstMorning, partOfData, bias, predictedFourthRoom)
    # ownDsM2, ownFiveFractileM2 = NNOwnDataSet(locations, fileSecondMorning, partOfData, bias, predictedFourthRoom)
    # ownDsM3, ownFiveFractileM3 = NNOwnDataSet(locations, fileThirdMorning, partOfData, bias, predictedFourthRoom)
    # ownDsM4, ownFiveFractileM4 = NNOwnDataSet(locations, fileFourthMorning, partOfData, bias, predictedFourthRoom)

    # predictions.append([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    # ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    # ownFiveFractileM = (ownFiveFractileM1 + ownFiveFractileM2 + ownFiveFractileM3 + ownFiveFractileM4) / 4
    
    # """ Evening own dataset """
    
    # ownDsE1, ownFiveFractileE1 = NNOwnDataSet(locations, fileFirstEvening, partOfData, bias, predictedFourthRoom)
    # ownDsE2, ownFiveFractileE2 = NNOwnDataSet(locations, fileSecondEvening, partOfData, bias, predictedFourthRoom)
    # ownDsE3, ownFiveFractileE3 = NNOwnDataSet(locations, fileThirdEvening, partOfData, bias, predictedFourthRoom)

    # predictions.append([ownDsE1, ownDsE2, ownDsE3])
    # ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    # ownFiveFractileE = (ownFiveFractileE1 + ownFiveFractileE2 + ownFiveFractileE3) / 3
    
    # """ Morning other dataset """
    
    # otherDayM11, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM12, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM13, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM21, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM22, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM23, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM31, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM32, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM33, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM41, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM42, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherDayM43, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    
    # predictions.append([otherDayM11, otherDayM12, otherDayM13, otherDayM21, otherDayM22, otherDayM23, otherDayM31, otherDayM32, otherDayM33, otherDayM41, otherDayM42, otherDayM43])
    # otherDayM = (otherDayM11 + otherDayM12 + otherDayM13 + otherDayM21 + otherDayM22 + otherDayM23 + otherDayM31 + otherDayM32 + otherDayM33 + otherDayM41 + otherDayM42 + otherDayM43) / 12
    # otherDayFiveFractileM = (otherDayFiveFractileM1 + otherDayFiveFractileM2 + otherDayFiveFractileM3 + otherDayFiveFractileM4) / 4
    
    # """ Morning other time """
    
    # otherTimeM11, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM12, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM13, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM21, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM22, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM23, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM31, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM32, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM33, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM41, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM42, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    # otherTimeM43, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    
    # predictions.append([otherTimeM11, otherTimeM12, otherTimeM13, otherTimeM21, otherTimeM22, otherTimeM23, otherTimeM31, otherTimeM32, otherTimeM33, otherTimeM41, otherTimeM42, otherTimeM43])
    # otherTimeM = (otherTimeM11 + otherTimeM12 + otherTimeM13 + otherTimeM21 + otherTimeM22 + otherTimeM23 + otherTimeM31 + otherTimeM32 + otherTimeM33 + otherTimeM41 + otherTimeM42 + otherTimeM43) / 12
    # otherTimeFiveFractileM = (otherTimeFiveFractileM1 + otherTimeFiveFractileM2 + otherTimeFiveFractileM3 + otherTimeFiveFractileM4) / 4
    
    # """ Evening other dataset """
    
    # otherDayE11, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    # otherDayE12, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    # otherDayE21, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherDayE22, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    # otherDayE31, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    # otherDayE32, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    
    # predictions.append([otherDayE11, otherDayE12, otherDayE21, otherDayE22, otherDayE31, otherDayE32])
    # otherDayE = (otherDayE11 + otherDayE12 + otherDayE21 + otherDayE22 + otherDayE31 + otherDayE32) / 6
    # otherDayFiveFractileE = (otherDayFiveFractileE1 + otherDayFiveFractileE2 + otherDayFiveFractileE3) / 3
    
    # """ Evening other time """
    
    # otherTimeE11, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE12, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE13, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE14, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE21, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE22, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE23, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE24, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE31, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE32, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE33, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    # otherTimeE34, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    
    # predictions.append([otherTimeE11, otherTimeE12, otherTimeE13, otherTimeE14, otherTimeE21, otherTimeE22, otherTimeE23, otherTimeE24, otherTimeE31, otherTimeE32, otherTimeE33, otherTimeE34])
    # otherTimeE = (otherTimeE11 + otherTimeE12 + otherTimeE13 + otherTimeE14 + otherTimeE21 + otherTimeE22 + otherTimeE23 + otherTimeE24 + otherTimeE31 + otherTimeE32 + otherTimeE33 + otherTimeE34) / 12
    # otherTimeFiveFractileE = (otherTimeFiveFractileE1 + otherTimeFiveFractileE2 + otherTimeFiveFractileE3) / 3
    
    # return ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE, predictions
    

# def changePredictions(predictions):
#     predictionsUpdated = []
#     for predict in predictions:
#         predictionsUpdated.append(predict)
#     return predictionsUpdated

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