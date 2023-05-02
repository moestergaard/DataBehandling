from SupportVectorMachine import SVMOwnDataSet, SVMAgainstOtherDatasets
from NeuralNetwork import NNOwnDataSet, NNAgainstOtherDatasets

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    locationsFull = ["Kontor", "Stue", "Køkken", "Entré"]
    
    fileFirstMorning = "Data/WifiData230418_9-12.txt"
    fileSecondMorning = "Data/WifiData230420_9-12.txt"
    fileThirdMorning = "Data/WifiData230421_9-12.txt"
    fileFourthMorning = "Data/WifiData230424_9-12.txt"
    fileFirstEvening = "Data/WifiData230421_17-21.txt"
    fileSecondEvening = "Data/WifiData230423_17-21.txt"
    fileThirdEvening = "Data/WifiData230424_17-21.txt"
    
    # partOfData = [1, 2/3, 1/3, 2/9, 1/9]
    # minutes = [45, 30, 15, 10, 5]
    partOfData = [1]
    minutes = [45]
    
    """ SVM three rooms """
    
    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i])
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM3")
    
    """ SVM four rooms """
    
    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i])
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM4")
        
    # """ NN three rooms with bias """

    # for i in range(len(partOfData)):
    #     ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True, False)
    #     printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    # """ NN three rooms with bias - predicts fourth room """

    # for i in range(len(partOfData)):
    #     ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True, True)
    #     printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B-Predicts", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    # """ NN three rooms without bias """

    # for i in range(len(partOfData)):
    #     ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False, False)
    #     printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
    # """ NN three rooms without bias - predicts fourth room """

    # for i in range(len(partOfData)):
    #     ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False, True)
    #     printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB-Predicts", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
    """ NN four rooms without bias """

    for i in range(len(partOfData)):
        ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i], False, False)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
def predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = 0
    
    """ Morning own dataset """
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, partOfData)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, partOfData)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, partOfData)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, partOfData)

    print([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1 = SVMOwnDataSet(locations, fileFirstEvening, partOfData)
    ownDsE2 = SVMOwnDataSet(locations, fileSecondEvening, partOfData)
    ownDsE3 = SVMOwnDataSet(locations, fileThirdEvening, partOfData)

    print([ownDsE1, ownDsE2, ownDsE3])
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
    
    print([otherDayM11, otherDayM12, otherDayM13, otherDayM21, otherDayM22, otherDayM23, otherDayM31, otherDayM32, otherDayM33, otherDayM41, otherDayM42, otherDayM43])
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
    
    print([otherTimeM11, otherTimeM12, otherTimeM13, otherTimeM21, otherTimeM22, otherTimeM23, otherTimeM31, otherTimeM32, otherTimeM33, otherTimeM41, otherTimeM42, otherTimeM43])
    otherTimeM = (otherTimeM11 + otherTimeM12 + otherTimeM13 + otherTimeM21 + otherTimeM22 + otherTimeM23 + otherTimeM31 + otherTimeM32 + otherTimeM33 + otherTimeM41 + otherTimeM42 + otherTimeM43) / 12
    
    """ Evening other dataset """
    
    otherDayE11 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening], partOfData)
    otherDayE12 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdEvening], partOfData)
    otherDayE21 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening], partOfData)
    otherDayE22 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdEvening], partOfData)
    otherDayE31 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening], partOfData)
    otherDayE32 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondEvening], partOfData)
    
    print([otherDayE11, otherDayE12, otherDayE21, otherDayE22, otherDayE31, otherDayE32])
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
    
    print([otherTimeE11, otherTimeE12, otherTimeE13, otherTimeE14, otherTimeE21, otherTimeE22, otherTimeE23, otherTimeE24, otherTimeE31, otherTimeE32, otherTimeE33, otherTimeE34])
    otherTimeE = (otherTimeE11 + otherTimeE12 + otherTimeE13 + otherTimeE14 + otherTimeE21 + otherTimeE22 + otherTimeE23 + otherTimeE24 + otherTimeE31 + otherTimeE32 + otherTimeE33 + otherTimeE34) / 12
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE

def predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData, bias, predictedFourthRoom):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = ownFiveFractileM = ownFiveFractileE = 0
    
    """ Morning own dataset """
    
    ownDsM1, ownFiveFractileM1 = NNOwnDataSet(locations, fileFirstMorning, partOfData, bias, predictedFourthRoom)
    ownDsM2, ownFiveFractileM2 = NNOwnDataSet(locations, fileSecondMorning, partOfData, bias, predictedFourthRoom)
    ownDsM3, ownFiveFractileM3 = NNOwnDataSet(locations, fileThirdMorning, partOfData, bias, predictedFourthRoom)
    ownDsM4, ownFiveFractileM4 = NNOwnDataSet(locations, fileFourthMorning, partOfData, bias, predictedFourthRoom)

    print([ownDsM1, ownDsM2, ownDsM3, ownDsM4])
    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    ownFiveFractileM = (ownFiveFractileM1 + ownFiveFractileM2 + ownFiveFractileM3 + ownFiveFractileM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1, ownFiveFractileE1 = NNOwnDataSet(locations, fileFirstEvening, partOfData, bias, predictedFourthRoom)
    ownDsE2, ownFiveFractileE2 = NNOwnDataSet(locations, fileSecondEvening, partOfData, bias, predictedFourthRoom)
    ownDsE3, ownFiveFractileE3 = NNOwnDataSet(locations, fileThirdEvening, partOfData, bias, predictedFourthRoom)

    print([ownDsE1, ownDsE2, ownDsE3])
    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    ownFiveFractileE = (ownFiveFractileE1 + ownFiveFractileE2 + ownFiveFractileE3) / 3
    
    """ Morning other dataset """
    
    otherDayM11, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherDayM12, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    otherDayM13, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    otherDayM21, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherDayM22, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    otherDayM23, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    otherDayM31, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherDayM32, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherDayM33, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    otherDayM41, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherDayM42, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherDayM43, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    
    print([otherDayM11, otherDayM12, otherDayM13, otherDayM21, otherDayM22, otherDayM23, otherDayM31, otherDayM32, otherDayM33, otherDayM41, otherDayM42, otherDayM43])
    otherDayM = (otherDayM11 + otherDayM12 + otherDayM13 + otherDayM21 + otherDayM22 + otherDayM23 + otherDayM31 + otherDayM32 + otherDayM33 + otherDayM41 + otherDayM42 + otherDayM43) / 12
    otherDayFiveFractileM = (otherDayFiveFractileM1 + otherDayFiveFractileM2 + otherDayFiveFractileM3 + otherDayFiveFractileM4) / 4
    
    """ Morning other time """
    
    otherTimeM11, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM12, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM13, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM21, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM22, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM23, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM31, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM32, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM33, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM41, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM42, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    otherTimeM43, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    
    print([otherTimeM11, otherTimeM12, otherTimeM13, otherTimeM21, otherTimeM22, otherTimeM23, otherTimeM31, otherTimeM32, otherTimeM33, otherTimeM41, otherTimeM42, otherTimeM43])
    otherTimeM = (otherTimeM11 + otherTimeM12 + otherTimeM13 + otherTimeM21 + otherTimeM22 + otherTimeM23 + otherTimeM31 + otherTimeM32 + otherTimeM33 + otherTimeM41 + otherTimeM42 + otherTimeM43) / 12
    otherTimeFiveFractileM = (otherTimeFiveFractileM1 + otherTimeFiveFractileM2 + otherTimeFiveFractileM3 + otherTimeFiveFractileM4) / 4
    
    """ Evening other dataset """
    
    otherDayE11, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    otherDayE12, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    otherDayE21, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherDayE22, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdEvening], partOfData, bias, predictedFourthRoom)
    otherDayE31, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening], partOfData, bias, predictedFourthRoom)
    otherDayE32, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondEvening], partOfData, bias, predictedFourthRoom)
    
    print([otherDayE11, otherDayE12, otherDayE21, otherDayE22, otherDayE31, otherDayE32])
    otherDayE = (otherDayE11 + otherDayE12 + otherDayE21 + otherDayE22 + otherDayE31 + otherDayE32) / 6
    otherDayFiveFractileE = (otherDayFiveFractileE1 + otherDayFiveFractileE2 + otherDayFiveFractileE3) / 3
    
    """ Evening other time """
    
    otherTimeE11, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE12, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE13, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE14, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE21, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE22, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE23, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE24, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE31, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE32, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileSecondMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE33, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileThirdMorning], partOfData, bias, predictedFourthRoom)
    otherTimeE34, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFourthMorning], partOfData, bias, predictedFourthRoom)
    
    print([otherTimeE11, otherTimeE12, otherTimeE13, otherTimeE14, otherTimeE21, otherTimeE22, otherTimeE23, otherTimeE24, otherTimeE31, otherTimeE32, otherTimeE33, otherTimeE34])
    otherTimeE = (otherTimeE11 + otherTimeE12 + otherTimeE13 + otherTimeE14 + otherTimeE21 + otherTimeE22 + otherTimeE23 + otherTimeE24 + otherTimeE31 + otherTimeE32 + otherTimeE33 + otherTimeE34) / 12
    otherTimeFiveFractileE = (otherTimeFiveFractileE1 + otherTimeFiveFractileE2 + otherTimeFiveFractileE3) / 3
    
    return ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE
    

def printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes, modelType, ownFiveFractileM = 0, ownFiveFractileE = 0, otherDayFiveFractileM = 0, otherDayFiveFractileE = 0, otherTimeFiveFractileM = 0, otherTimeFiveFractileE = 0):
    print(f"****************** {modelType} ******************")
    print("Antal minutter: ", minutes)
    print()
    print("Nøjagtighed egen dataindsamling")
    print("Morgen: %2.2f procent" % (ownDsM*100))
    if (ownFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (ownFiveFractileM*100))
    print("Aften: %2.2f procent" % (ownDsE*100))
    if (ownFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (ownFiveFractileE*100))
    print()
    print("Nøjagtighed anden dag")
    print("Morgen: %2.2f procent" % (otherDayM*100))
    if (otherDayFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (otherDayFiveFractileM*100))
    print("Aften: %2.2f procent" % (otherDayE*100))
    if (otherDayFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (otherDayFiveFractileE*100))
    print()
    print("Nøjagtighed andet tidspunkt")
    print("Morgen: %2.2f procent" % (otherTimeM*100))
    if (otherTimeFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (otherTimeFiveFractileM*100))
    print("Aften: %2.2f procent" % (otherTimeE*100))
    if (otherTimeFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (otherTimeFiveFractileE*100))
    print()
    
if __name__ == '__main__':
    main()