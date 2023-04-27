from SupportVectorMachine import SVMOwnDataSet, SVMAgainstOtherDatasets
from TestingNeuralNetwork import DataSet, smallDataSetTestedAgainstBigDataSet

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
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], True)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-B")
    
    """ NN three rooms without bias """

    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i], False)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN3-UB")
        
    """ NN four rooms without bias """

    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, partOfData[i], False)
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "NN4-UB")
    
def predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfFile):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = 0
    
    """ Morning own dataset """
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, partOfFile)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, partOfFile)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, partOfFile)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, partOfFile)

    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1 = SVMOwnDataSet(locations, fileFirstEvening, partOfFile)
    ownDsE2 = SVMOwnDataSet(locations, fileSecondEvening, partOfFile)
    ownDsE3 = SVMOwnDataSet(locations, fileThirdEvening, partOfFile)

    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    
    """ Morning other dataset """
    
    otherDayM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherDayM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherDayM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning, fileSecondMorning, fileFourthMorning], partOfFile)
    otherDayM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning, fileSecondMorning, fileThirdMorning], partOfFile)
    
    otherDayM = (otherDayM1 + otherDayM2 + otherDayM3 + otherDayM4) / 4
    
    """ Evening other dataset """
    
    otherDayE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening, fileThirdEvening], partOfFile)
    otherDayE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening, fileThirdEvening], partOfFile)
    otherDayE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening, fileSecondEvening], partOfFile)
    
    otherDayE = (otherDayE1 + otherDayE2 + otherDayE3) / 3
    
    """ Morning other time """
    
    otherTimeM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    
    otherTimeM = (otherTimeM1 + otherTimeM2 + otherTimeM3 + otherTimeM4) / 4
    
    """ Evening other time """
    
    otherTimeE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    
    otherTimeE = (otherTimeE1 + otherTimeE2 + otherTimeE3) / 3
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE

def predictionsNN(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfFile, bias):
    ownDsM = ownDsE = otherDayM = otherDayE = otherTimeM = otherTimeE = 0
    
    """ Morning own dataset """
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, partOfFile)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, partOfFile)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, partOfFile)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, partOfFile)

    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    """ Evening own dataset """
    
    ownDsE1 = SVMOwnDataSet(locations, fileFirstEvening, partOfFile)
    ownDsE2 = SVMOwnDataSet(locations, fileSecondEvening, partOfFile)
    ownDsE3 = SVMOwnDataSet(locations, fileThirdEvening, partOfFile)

    ownDsE = (ownDsE1 + ownDsE2 + ownDsE3) / 3
    
    """ Morning other dataset """
    
    otherDayM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherDayM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherDayM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstMorning, fileSecondMorning, fileFourthMorning], partOfFile)
    otherDayM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstMorning, fileSecondMorning, fileThirdMorning], partOfFile)
    
    otherDayM = (otherDayM1 + otherDayM2 + otherDayM3 + otherDayM4) / 4
    
    """ Evening other dataset """
    
    otherDayE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening, fileThirdEvening], partOfFile)
    otherDayE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening, fileThirdEvening], partOfFile)
    otherDayE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening, fileSecondEvening], partOfFile)
    
    otherDayE = (otherDayE1 + otherDayE2 + otherDayE3) / 3
    
    """ Morning other time """
    
    otherTimeM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    
    otherTimeM = (otherTimeM1 + otherTimeM2 + otherTimeM3 + otherTimeM4) / 4
    
    """ Evening other time """
    
    otherTimeE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    
    otherTimeE = (otherTimeE1 + otherTimeE2 + otherTimeE3) / 3
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE
    

def printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes, modelType):
    print(f"****************** {modelType} ******************")
    print("Antal minutter: ", minutes)
    print()
    print("Nøjagtighed egen dataindsamling")
    print("Morgen: %2.2f procent" % (ownDsM*100))
    print("Aften: %2.2f procent" % (ownDsE*100))
    print()
    print("Nøjagtighed anden dag")
    print("Morgen: %2.2f procent" % (otherDayM*100))
    print("Aften: %2.2f procent" % (otherDayE*100))
    print()
    print("Nøjagtighed andet tidspunkt")
    print("Morgen: %2.2f procent" % (otherTimeM*100))
    print("Aften: %2.2f procent" % (otherTimeE*100))
    print()
    
if __name__ == '__main__':
    main()