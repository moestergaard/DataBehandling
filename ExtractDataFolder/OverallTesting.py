from SupportVectorMachine import SVMOwnDataSet, SVMAgainstOtherDatasets
from TestingNeuralNetwork import NNOwnDataSet, NNAgainstOtherDatasets

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    locationsFull = ["Kontor", "Stue", "Køkken", "Entré"]
    
    fileFirstMorning = "WifiData230418_9-12.txt"
    fileSecondMorning = "WifiData230420_9-12.txt"
    fileThirdMorning = "WifiData230421_9-12.txt"
    fileFourthMorning = "WifiData230424_9-12.txt"
    fileFirstEvening = "WifiData230421_17-21.txt"
    fileSecondEvening = "WifiData230423_17-21.txt"
    fileThirdEvening = "WifiData230424_17-21.txt"
    
    partOfData = [1, 2/3, 1/3, 2/9, 1/9]
    minutes = [45, 30, 15, 10, 5]
    
    """ SVM three rooms """
    
    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i])
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM3")
    
    """ SVM four rooms """
    
    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i])
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM4")
        
    """ NN three rooms with bias """

    for i in range(len(partOfData)):
        ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
    """ NN three rooms without bias """

    for i in range(len(partOfData)):
        ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
        
    """ NN four rooms without bias """

    for i in range(len(partOfData)):
        ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i], False)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB", ownFiveFractileM, ownFiveFractileE, otherDayFiveFractileM, otherDayFiveFractileE, otherTimeFiveFractileM, otherTimeFiveFractileE)
    
def predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = 0
    
    """ Morning own dataset """
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, partOfData)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, partOfData)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, partOfData)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, partOfData)

    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1 = SVMOwnDataSet(locations, fileFirstEvening, partOfData)
    ownDsE2 = SVMOwnDataSet(locations, fileSecondEvening, partOfData)
    ownDsE3 = SVMOwnDataSet(locations, fileThirdEvening, partOfData)

    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    
    """ Morning other dataset """
    
    otherDayM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData)
    otherDayM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning, fileThirdMorning, fileFourthMorning], partOfData)
    otherDayM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning, fileSecondMorning, fileFourthMorning], partOfData)
    otherDayM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning, fileSecondMorning, fileThirdMorning], partOfData)
    
    otherDayM = (otherDayM1 + otherDayM2 + otherDayM3 + otherDayM4) / 4
    
    """ Evening other dataset """
    
    otherDayE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening, fileThirdEvening], partOfData)
    otherDayE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening, fileThirdEvening], partOfData)
    otherDayE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening, fileSecondEvening], partOfData)
    
    otherDayE = (otherDayE1 + otherDayE2 + otherDayE3) / 3
    
    """ Morning other time """
    
    otherTimeM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData)
    otherTimeM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData)
    otherTimeM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData)
    otherTimeM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData)
    
    otherTimeM = (otherTimeM1 + otherTimeM2 + otherTimeM3 + otherTimeM4) / 4
    
    """ Evening other time """
    
    otherTimeE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData)
    otherTimeE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData)
    otherTimeE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData)
    
    otherTimeE = (otherTimeE1 + otherTimeE2 + otherTimeE3) / 3
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE

def predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData, bias):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = ownFiveFractileM = ownFiveFractileE = 0
    
    """ Morning own dataset """
    
    ownDsM1, ownFiveFractileM1 = NNOwnDataSet(locations, fileFirstMorning, partOfData, bias)
    ownDsM2, ownFiveFractileM2 = NNOwnDataSet(locations, fileSecondMorning, partOfData, bias)
    ownDsM3, ownFiveFractileM3 = NNOwnDataSet(locations, fileThirdMorning, partOfData, bias)
    ownDsM4, ownFiveFractileM4 = NNOwnDataSet(locations, fileFourthMorning, partOfData, bias)

    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    ownFiveFractileM = (ownFiveFractileM1 + ownFiveFractileM2 + ownFiveFractileM3 + ownFiveFractileM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1, ownFiveFractileE1 = NNOwnDataSet(locations, fileFirstEvening, partOfData, bias)
    ownDsE2, ownFiveFractileE2 = NNOwnDataSet(locations, fileSecondEvening, partOfData, bias)
    ownDsE3, ownFiveFractileE3 = NNOwnDataSet(locations, fileThirdEvening, partOfData, bias)

    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    ownFiveFractileE = (ownFiveFractileE1 + ownFiveFractileE2 + ownFiveFractileE3) / 3
    
    """ Morning other dataset """
    
    otherDayM1, otherDayFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData, bias)
    otherDayM2, otherDayFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning, fileThirdMorning, fileFourthMorning], partOfData, bias)
    otherDayM3, otherDayFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning, fileSecondMorning, fileFourthMorning], partOfData, bias)
    otherDayM4, otherDayFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning, fileSecondMorning, fileThirdMorning], partOfData, bias)
    
    otherDayM = (otherDayM1 + otherDayM2 + otherDayM3 + otherDayM4) / 4
    otherDayFiveFractileM = (otherDayFiveFractileM1 + otherDayFiveFractileM2 + otherDayFiveFractileM3 + otherDayFiveFractileM4) / 4
    
    """ Evening other dataset """
    
    otherDayE1, otherDayFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening, fileThirdEvening], partOfData, bias)
    otherDayE2, otherDayFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening, fileThirdEvening], partOfData, bias)
    otherDayE3, otherDayFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening, fileSecondEvening], partOfData, bias)
    
    otherDayE = (otherDayE1 + otherDayE2 + otherDayE3) / 3
    otherDayFiveFractileE = (otherDayFiveFractileE1 + otherDayFiveFractileE2 + otherDayFiveFractileE3) / 3
    
    """ Morning other time """
    
    otherTimeM1, otherTimeFiveFractileM1 = NNAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData, bias)
    otherTimeM2, otherTimeFiveFractileM2 = NNAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData, bias)
    otherTimeM3, otherTimeFiveFractileM3 = NNAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData, bias)
    otherTimeM4, otherTimeFiveFractileM4 = NNAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfData, bias)
    
    otherTimeM = (otherTimeM1 + otherTimeM2 + otherTimeM3 + otherTimeM4) / 4
    otherTimeFiveFractileM = (otherTimeFiveFractileM1 + otherTimeFiveFractileM2 + otherTimeFiveFractileM3 + otherTimeFiveFractileM4) / 4
    
    """ Evening other time """
    
    otherTimeE1, otherTimeFiveFractileE1 = NNAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData, bias)
    otherTimeE2, otherTimeFiveFractileE2 = NNAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData, bias)
    otherTimeE3, otherTimeFiveFractileE3 = NNAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfData, bias)
    
    otherTimeE = (otherTimeE1 + otherTimeE2 + otherTimeE3) / 3
    otherTimeFiveFractileE = (otherTimeFiveFractileE1 + otherTimeFiveFractileE2 + otherTimeFiveFractileE3) / 3
    
    return ownDsM, ownFiveFractileM, ownDsE, ownFiveFractileE, otherDayM, otherDayFiveFractileM, otherDayE, otherDayFiveFractileE, otherTimeM, otherTimeFiveFractileM, otherTimeE, otherTimeFiveFractileE
    

def printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes, modelType, ownFiveFractileM = 0, ownFiveFractileE = 0, otherDayFiveFractileM = 0, otherDayFiveFractileE = 0, otherTimeFiveFractileM = 0, otherTimeFiveFractileE = 0):
    print(f"****************** {modelType} ******************")
    print("Antal minutter: ", minutes)
    print()
    print("Nøjagtighed egen dataindsamling")
    print("Morgen: %2.2f procent" % (ownDsM*100))
    if (ownFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (ownFiveFractileM))
    print("Aften: %2.2f procent" % (ownDsE*100))
    if (ownFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (ownFiveFractileE))
    print()
    print("Nøjagtighed anden dag")
    print("Morgen: %2.2f procent" % (otherDayM*100))
    if (otherDayFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (otherDayFiveFractileM))
    print("Aften: %2.2f procent" % (otherDayE*100))
    if (otherDayFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (otherDayFiveFractileE))
    print()
    print("Nøjagtighed andet tidspunkt")
    print("Morgen: %2.2f procent" % (otherTimeM*100))
    if (otherTimeFiveFractileM != 0):
        print("5-fraktil morgen: %2.2f procent" % (otherTimeFiveFractileM))
    print("Aften: %2.2f procent" % (otherTimeE*100))
    if (otherTimeFiveFractileE != 0):
        print("5-fraktil aften: %2.2f procent" % (otherTimeFiveFractileE))
    print()
    
if __name__ == '__main__':
    main()