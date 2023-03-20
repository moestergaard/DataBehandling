import numpy as np
from sklearn import svm

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    groundFloor = ["Stue, Køkken"]
    firstFloor = ["Kontor"]

    bigDataSetSVM(locations)
    smallDataSetTestedAgainstBigDataSetSVM(locations)

    bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations)
    smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations)
    
    # changeDataFile(filename)
    
def bigDataSetSVMSeparateFloors(groundFloor, firstFloor, locations):
    filename = 'WifiData2303141637Modified.txt'
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)

    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)
    trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor = extractDataSeparateFloors(trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples)

    predictionSVM = calculationsSVM(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** BigDataSetSVMSeparateFloors ***")
    accuracySVM(labelsTestFloors, predictionSVM)

    predictionSVM = calculationsSVM(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** BigDataSetSVMSeparateFloors Ground Floor ***")
    accuracySVM(labelsTestGroundFloor, predictionSVM)

def smallDataSetTestedAgainstBigDataSetSVMSeparateFloors(locations):
    filename = 'WifiData2-2303172344.txt'
    filenameTest = 'WifiData2303141637Modified.txt'

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest)

    

    trainingSamples, trainingLabels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)
    
    testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSIDTest, dataPointsTest, locations)

    #
    # Removes all new BSSID that were not present at origianl syncronising
    #
    for i in range(0, len(distinctBSSIDTest)):
        for i in range(0, len(distinctBSSIDTest)):
            if not distinctBSSID.__contains__(distinctBSSIDTest[i]):
                testSamples = np.delete(testSamples, i, 1)
                distinctBSSIDTest.remove(distinctBSSIDTest[i])
                break

    #
    # Add rows of zeroes to testSample for all BSSID from syncroized data not present in tests
    #
    for i in range(0, len(distinctBSSID)):
        if not distinctBSSIDTest.__contains__(distinctBSSID[i]):
            distinctBSSIDTest.append(distinctBSSID[i])
            testSamples = np.append(testSamples, np.zeros((testSamples.shape[0],1)), 1)

    trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor = extractDataSeparateFloors(trainingSamples, trainingLabels, testSamples, testLabels)
                
    predictionSVM = calculationsSVM(trainingSamplesFloors, labelsTrainingFloors, testSamplesFloors)
    print("*** SmallDataSetTestedAgainstBigDataSetSVMSeparateFloors ***")
    accuracySVM(labelsTestFloors, predictionSVM)

    predictionSVM = calculationsSVM(trainingSamplesGroundFloor, labelsTrainingGroundFloor, testSamplesGroundFloor)
    print("*** SmallDataSetTestedAgainstBigDataSetSVMSeparateFloors Ground Floor ***")
    accuracySVM(labelsTestGroundFloor, predictionSVM)


def smallDataSetTestedAgainstBigDataSetSVM(locations):
    filename = 'WifiData2-2303172344.txt'
    filenameTest = 'WifiData2303141637Modified.txt'

    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    distinctBSSIDTest, dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest)

    

    trainingSamples, trainingLabels = extractDataCombined(filename, distinctBSSID, dataPoints, locations)

    testSamples, testLabels = extractDataCombined(filenameTest, distinctBSSIDTest, dataPointsTest, locations)

    #
    # Removes all new BSSID that were not present at origianl syncronising
    #
    for i in range(0, len(distinctBSSIDTest)):
        for i in range(0, len(distinctBSSIDTest)):
            if not distinctBSSID.__contains__(distinctBSSIDTest[i]):
                testSamples = np.delete(testSamples, i, 1)
                distinctBSSIDTest.remove(distinctBSSIDTest[i])
                break

    #
    # Add rows of zeroes to testSample for all BSSID from syncroized data not present in tests
    #
    for i in range(0, len(distinctBSSID)):
        if not distinctBSSIDTest.__contains__(distinctBSSID[i]):
            distinctBSSIDTest.append(distinctBSSID[i])
            testSamples = np.append(testSamples, np.zeros((testSamples.shape[0],1)), 1)
                
    predictionSVM = calculationsSVM(trainingSamples, trainingLabels, testSamples)

    print("*** SmallDataSetTestedAgainstBigDataSetSVM ***")
    accuracySVM(testLabels, predictionSVM)


def bigDataSetSVM(locations):
    filename = 'WifiData2303141637Modified.txt'
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    predictionSVM = calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples)
    print("*** BigDataSetSVM ***")
    accuracySVM(labelsTestSamples, predictionSVM)


#
# Makes an array consisting of the distinct BSSID.
#
def extractDistinctBSSIDAndNumberOfDataPoints(filename):
    distinctBSSID = []
    dataPoints = 0

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("BSSID"):
                bssid = line.split(": ")[1].strip()

                alreadyIncludedInDistinctBSSID = distinctBSSID.__contains__(bssid)
                if not alreadyIncludedInDistinctBSSID:
                    distinctBSSID.append(bssid)
            if line.__contains__("Scanning"):
                dataPoints += 1

    return distinctBSSID, dataPoints

#
# Makes the following matrices
#   For training: m is the number of training samples, and n is the number of features (distinct BSSID)
#   A mxn matrice
#   A mx1 matrice:  The labels corresponding to training samples.
#  
#   For testing: k is the number of test samples, and n is the same as before.
#   A kxn matrice
#   A kx1 matrice:  The labels corresponding to test samples.
#
def extractData(filename, distinctBSSID, samples, locations):

    numberOfFeatures = len(distinctBSSID)
    numberOfTestSamples = np.ceil(samples/5).astype(int)
    numberOfTrainingSamples = samples - numberOfTestSamples

    trainingSamples = np.zeros((numberOfTrainingSamples, numberOfFeatures))
    labelsTrainingSamples = np.zeros((numberOfTrainingSamples, ))
    
    testSamples = np.zeros((numberOfTestSamples, numberOfFeatures))
    labelsTestSamples = np.zeros((numberOfTestSamples, ))

    twentyPercentTestData = 4
    indexTestSample = -1
    indexTrainingSample = -1

    currentBSSID = ""
    

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"):
                twentyPercentTestData = np.mod(twentyPercentTestData + 1, 5)
            if line.__contains__("BSSID"):
                currentBSSID = line.split(": ")[1].split()
            if line.__contains__("Scanning"):
                    location = line.split(": ")[1].strip()
                    for i in range(0, len(locations)):
                        if locations[i].__eq__(location):
                            if twentyPercentTestData == 4:
                                indexTestSample += 1
                                labelsTestSamples[indexTestSample] = i
                            else:
                                indexTrainingSample += 1
                                labelsTrainingSamples[indexTrainingSample] = i
            if line.__contains__("ResultLevel"):
                resultLevel = line.split(": ")[1].split()
                if twentyPercentTestData == 4:
                    testSamples = changeMatrice(testSamples, indexTestSample, distinctBSSID, currentBSSID[0], resultLevel[0])
                else: trainingSamples = changeMatrice(trainingSamples, indexTrainingSample, distinctBSSID, currentBSSID[0], resultLevel[0])

    return trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples

def extractDataSeparateFloors(trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples):

    trainingSamplesFloors = np.zeros((0, trainingSamples.shape[1]))
    trainingSamplesGroundFloor = np.zeros((0, trainingSamples.shape[1]))
    labelsTrainingFloors = np.zeros((0, ))
    labelsTrainingGroundFloor = np.zeros((0, ))

    testSamplesFloors = np.zeros((0, testSamples.shape[1]))
    testSamplesGroundFloor = np.zeros((0, testSamples.shape[1]))
    labelsTestFloors = np.zeros((0, ))
    labelsTestGroundFloor = np.zeros((0, ))

    for i in range(0, trainingSamples.shape[0]):
        trainingSamplesFloors = np.vstack((trainingSamplesFloors, trainingSamples[i]))
        if labelsTrainingSamples[i] == 0:
            labelsTrainingFloors = np.append(labelsTrainingFloors, labelsTrainingSamples[i])
        else:
            trainingSamplesGroundFloor = np.vstack((trainingSamplesGroundFloor, trainingSamples[i]))
            labelsTrainingGroundFloor = np.append(labelsTrainingGroundFloor, labelsTrainingSamples[i])

            labelsTrainingFloors = np.append(labelsTrainingFloors, 1)


    for i in range(0, testSamples.shape[0]):
        testSamplesFloors = np.vstack((testSamplesFloors, testSamples[i]))
        if labelsTestSamples[i] == 0:
            labelsTestFloors = np.append(labelsTestFloors, labelsTestSamples[i])
        else:
            testSamplesGroundFloor = np.vstack((testSamplesGroundFloor, testSamples[i]))
            labelsTestGroundFloor = np.append(labelsTestGroundFloor, labelsTestSamples[i])

            labelsTestFloors = np.append(labelsTestFloors, 1)



    return trainingSamplesFloors, trainingSamplesGroundFloor, labelsTrainingFloors, labelsTrainingGroundFloor, testSamplesFloors, testSamplesGroundFloor, labelsTestFloors,labelsTestGroundFloor

def extractDataCombined(filename, distinctBSSID, numberOfSamples, locations):

    samples = np.zeros((numberOfSamples, len(distinctBSSID)))
    labels = np.zeros((numberOfSamples, ))
    
    index = 0

    currentBSSID = ""

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"): 
                index += 1
            if line.__contains__("BSSID"):
                currentBSSID = line.split(": ")[1].split()
            if line.__contains__("Scanning"):
                    location = line.split(": ")[1].strip()
                    for i in range(0, len(locations)):
                        if locations[i].__eq__(location): labels[index] = i
            if line.__contains__("ResultLevel"):
                resultLevel = line.split(": ")[1].split()
                samples = changeMatrice(samples, index, distinctBSSID, currentBSSID[0], resultLevel[0])

    return samples, labels

def changeMatrice(matrice, index, distinctBSSID, currentBSSID, resultLevel):
    for i in range(0, len(distinctBSSID)):
        if distinctBSSID[i] == currentBSSID:
            matrice[index, i] = resultLevel
            return matrice    

def calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples):
    #clf = svm.SVC(kernel='poly', degree=3, C=1)
    #clf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1)
    
    clf = svm.SVC(C = 1, cache_size = 1000, class_weight='balanced')
    clf.fit(trainingSamples, labelsTrainingSamples)
    prediction = clf.predict(testSamples)

    return prediction

    
def accuracySVM(labelsTestSamples, predictionSVM):
    correct = 0
    wrong = 0

    shouldBeKitchenPredictsLivingRoom = 0
    shouldBeKitchenPredictsOffice = 0
    shouldBeLivingRoomPredictsKitchen = 0
    shouldBeLivingRoomPredictsOffice = 0
    shouldBeOfficePredictsKitchen = 0
    shouldBeOfficePredictsLivingRoom = 0

    for i in range(0, len(labelsTestSamples)):
        if labelsTestSamples[i] == predictionSVM[i]:
            correct += 1
        else:
            wrong += 1
            if labelsTestSamples[i] == 2:
                if predictionSVM[i] == 0: shouldBeKitchenPredictsOffice += 1
                else: shouldBeKitchenPredictsLivingRoom += 1
            elif labelsTestSamples[i] == 1:
                if predictionSVM[i] == 0: shouldBeLivingRoomPredictsOffice += 1
                else: shouldBeLivingRoomPredictsKitchen += 1
            else: 
                if predictionSVM[i] == 1: shouldBeOfficePredictsLivingRoom += 1
                else: shouldBeOfficePredictsKitchen += 1

    accuracy = correct/len(labelsTestSamples)
    print()
    print("********************************************************************************************")
    print()
    print("RESULT SUPPORT VECTOR MACHINE")
    print()
    print("Overall accuracy SVM is %2.2f percentage of %d tested data points." % (accuracy*100, len(labelsTestSamples)))
    print()
    print()
    print("Details for the wrong predictions")
    print()
    print("Wrong predicitions in total: ", wrong)
    print("Should be office but predicted kitchen %d corresponds to %2.2f percentage of wrongs." % (shouldBeOfficePredictsKitchen, shouldBeOfficePredictsKitchen/wrong*100))
    print("Should be office but predicted living room %d corresponds to %2.2f percentage of wrongs." % (shouldBeOfficePredictsLivingRoom, shouldBeOfficePredictsLivingRoom/wrong*100))
    print("Should be kitchen but predicted office %d corresponds to %2.2f percentage of wrongs." % (shouldBeKitchenPredictsOffice, shouldBeKitchenPredictsOffice/wrong*100))
    print("Should be kitchen but predicted living room %d corresponds to %2.2f percentage of wrongs." % (shouldBeKitchenPredictsLivingRoom, shouldBeKitchenPredictsLivingRoom/wrong*100))
    print("Should be living room but predicted office %d corresponds to %2.2f percentage of wrongs." % (shouldBeLivingRoomPredictsOffice, shouldBeLivingRoomPredictsOffice/wrong*100))
    print("Should be living room but predicted kitchen %d corresponds to %2.2f percentage of wrongs." % (shouldBeLivingRoomPredictsKitchen, shouldBeLivingRoomPredictsKitchen/wrong*100))
    print()
    print("********************************************************************************************")

def changeDataFile(filename):
    bssid = ""
    resultlevel = ""

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("Scanning"):
                print("**************")
                print()
            if line.__contains__("BSSID"):
                bssid = line.strip()
            elif line.__contains__("ResultLevel"):
                resultlevel = line.strip()
            elif line.__contains__("Frequency"):
                bssid += "+"
                bssid += line.split(": ")[1].split()[0]
                print(bssid)
                print(resultlevel)
                print(line.strip())
            else: print(line.strip())

if __name__ == '__main__':
    main()