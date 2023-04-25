import numpy as np
from sklearn import svm
from TestingSupportVectorMachine import SVMOwnDataSet, datasetTestedAgainstAnotherDatasetSVM, SVMAgainstOtherDatasets
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
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, 1)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 45, "SVM3")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, 2/3)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 30, "SVM3")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, 1/3)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 15, "SVM3")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, 2/9)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 10, "SVM3")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, 1/9)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 5, "SVM3")
    
    
    """ SVM four rooms """
    
    for i in range(len(partOfData)):
        ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations, partOfData[i])
        printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, minutes[i], "SVM4")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, 1)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 45, "SVM4")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, 2/3)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 30, "SVM4")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, 1/3)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 15, "SVM4")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, 2/9)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 10, "SVM4")
    
    # ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE = predictionsSVM(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull, 1/9)
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 5, "SVM4")
    
    
    # predictionsSVM3(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations)
    # predictionsSVM4(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locationsFull)
    
    # DataSet(locations, fileFirstMorning)
    # SVMOwnDataSet(locations, fileFirstMorning)
    # DataSet(locations, fileSecondMorning)
    # SVMOwnDataSet(locations, fileSecondMorning)
    
    # print()
    # print("*************** FIRE RUM ***************")
    # print()
    
    # DataSet(locationsFull, fileFirstMorning)
    # SVMOwnDataSet(locationsFull, fileFirstMorning)
    # DataSet(locationsFull, fileSecondMorning)
    # SVMOwnDataSet(locationsFull, fileSecondMorning)
    
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
    
    # otherDayM12 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileSecondMorning, partOfFile)
    # otherDayM13 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileThirdMorning, partOfFile)
    # otherDayM14 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileFourthMorning, partOfFile)
    # otherDayM21 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileFirstMorning, partOfFile)
    # otherDayM23 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileThirdMorning, partOfFile)
    # otherDayM24 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileFourthMorning, partOfFile)
    # otherDayM31 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileFirstMorning, partOfFile)
    # otherDayM32 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileSecondMorning, partOfFile)
    # otherDayM34 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileFourthMorning, partOfFile)
    # otherDayM41 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileFirstMorning, partOfFile)
    # otherDayM42 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileSecondMorning, partOfFile)
    # otherDayM43 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileThirdMorning, partOfFile)
    
    otherDayM = (otherDayM1 + otherDayM2 + otherDayM3 + otherDayM4) / 4
    
    """ Evening other dataset """
    
    otherDayE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileSecondEvening, fileThirdEvening], partOfFile)
    otherDayE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstEvening, fileThirdEvening], partOfFile)
    otherDayE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstEvening, fileSecondEvening], partOfFile)
    
    # otherDayE12 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileSecondEvening, partOfFile)
    # otherDayE13 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileThirdEvening, partOfFile)
    # otherDayE21 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileFirstEvening, partOfFile)
    # otherDayE23 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileThirdEvening, partOfFile)
    # otherDayE31 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileFirstEvening, partOfFile)
    # otherDayE32 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileSecondEvening, partOfFile)
    
    otherDayE = (otherDayE1 + otherDayE2 + otherDayE3) / 3
    
    """ Morning other time """
    
    otherTimeM1 = SVMAgainstOtherDatasets(locations, fileFirstMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM2 = SVMAgainstOtherDatasets(locations, fileSecondMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM3 = SVMAgainstOtherDatasets(locations, fileThirdMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    otherTimeM4 = SVMAgainstOtherDatasets(locations, fileFourthMorning, [fileFirstEvening, fileSecondEvening, fileThirdEvening], partOfFile)
    
    # otherTimeM11 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileFirstEvening, partOfFile)
    # otherTimeM12 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileSecondEvening, partOfFile)
    # otherTimeM13 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileThirdEvening, partOfFile)
    # otherTimeM21 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileFirstEvening, partOfFile)
    # otherTimeM22 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileSecondEvening, partOfFile)
    # otherTimeM23 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileThirdEvening, partOfFile)
    # otherTimeM31 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileFirstEvening, partOfFile)
    # otherTimeM32 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileSecondEvening, partOfFile)
    # otherTimeM33 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileThirdEvening, partOfFile)
    # otherTimeM41 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileFirstEvening, partOfFile)
    # otherTimeM42 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileSecondEvening, partOfFile)
    # otherTimeM43 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileThirdEvening, partOfFile)

    otherTimeM = (otherTimeM1 + otherTimeM2 + otherTimeM3 + otherTimeM4) / 4
    
    """ Evening other time """
    
    otherTimeE1 = SVMAgainstOtherDatasets(locations, fileFirstEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE2 = SVMAgainstOtherDatasets(locations, fileSecondEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    otherTimeE3 = SVMAgainstOtherDatasets(locations, fileThirdEvening, [fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning], partOfFile)
    
    # otherTimeE11 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileFirstMorning, partOfFile)
    # otherTimeE12 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileSecondMorning, partOfFile)
    # otherTimeE13 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileThirdMorning, partOfFile)
    # otherTimeE14 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstEvening, fileFourthMorning, partOfFile)
    # otherTimeE21 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileFirstMorning, partOfFile)
    # otherTimeE22 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileSecondMorning, partOfFile)
    # otherTimeE23 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileThirdMorning, partOfFile)
    # otherTimeE24 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondEvening, fileFourthMorning, partOfFile)
    # otherTimeE31 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileFirstMorning, partOfFile)
    # otherTimeE32 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileSecondMorning, partOfFile)
    # otherTimeE33 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileThirdMorning, partOfFile)
    # otherTimeE34 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdEvening, fileFourthMorning, partOfFile)
    
    otherTimeE = (otherTimeE1 + otherTimeE2 + otherTimeE3) / 3
    
    return ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE
    
    # printMethod(ownDsM, ownDsE, otherDayM, otherDayE, otherTimeM, otherTimeE, 45, "SVM3")
    
    # ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, np.ceil(5/45))
    # ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, np.ceil(5/45))
    # printMethod(ownDsM, ownDsE, otherDayM, OtherDayE, otherTimeM, otherTimeE, 5, "SVM3")
    
    
    
def predictionsSVM4(fileFirstMorning, fileSecondMorning, fileThirdMorning, fileFourthMorning, fileFirstEvening, fileSecondEvening, fileThirdEvening, locations):
    ownDsM = ownDsE = otherDayM = OtherDayE = otherTimeM = otherTimeE = 0
    
    ownDsM1 = SVMOwnDataSet(locations, fileFirstMorning, 1)
    ownDsM2 = SVMOwnDataSet(locations, fileSecondMorning, 1)
    ownDsM3 = SVMOwnDataSet(locations, fileThirdMorning, 1)
    ownDsM4 = SVMOwnDataSet(locations, fileFourthMorning, 1)
    # print(ownDsM3)
    ownDsM = (ownDsM1 + ownDsM2 + ownDsM3 + ownDsM4) / 4
    
    otherDayM12 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileSecondMorning, 1)
    otherDayM13 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileThirdMorning, 1)
    otherDayM14 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFirstMorning, fileFourthMorning, 1)
    otherDayM21 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileFirstMorning, 1)
    otherDayM23 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileThirdMorning, 1)
    otherDayM24 = datasetTestedAgainstAnotherDatasetSVM(locations, fileSecondMorning, fileFourthMorning, 1)
    otherDayM31 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileFirstMorning, 1)
    otherDayM32 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileSecondMorning, 1)
    otherDayM34 = datasetTestedAgainstAnotherDatasetSVM(locations, fileThirdMorning, fileFourthMorning, 1)
    otherDayM41 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileFirstMorning, 1)
    otherDayM42 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileSecondMorning, 1)
    otherDayM43 = datasetTestedAgainstAnotherDatasetSVM(locations, fileFourthMorning, fileThirdMorning, 1)
    
    otherDayM = (otherDayM12 + otherDayM13 + otherDayM14 + otherDayM21 + otherDayM23 + otherDayM24 + otherDayM31 + otherDayM32 + otherDayM34 + otherDayM41 + otherDayM42 + otherDayM43) / 12
    
    printMethod(ownDsM, ownDsE, otherDayM, OtherDayE, otherTimeM, otherTimeE, 45, "SVM4")






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