# from enum import Enum
import numpy as np
from sklearn import svm

# class Location(Enum):
#     Kontor = 1
#     Stue = 2
#     Køkken = 3

def main():
    filename = 'WifiData2303141637Modified.txt'
    locations = ["Kontor", "Stue", "Køkken"]
    # changeDataFile(filename)
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    print(distinctBSSID)
    print(len(distinctBSSID))
    print(dataPoints)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints, locations)

    predictionSVM = calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples)
    accuracySVM(labelsTestSamples, predictionSVM, locations)

    # clf = svm.SVC()
    # fitSVM(clf, trainingSamples, labelsTrainingSamples)
    # prediction = predictSVM(clf, testSamples)
    # accuracy = accuracySVM(prediction, labelsTestSamples)

    # print(trainingSamples)
    # print(labelsTrainingSamples)
    # print()
    # print(testSamples)
    # print(labelsTestSamples)


    

def changeDataFile(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("Scanning"):
                print("**************")
                print()
            print(line.strip())
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
    labelsTrainingSamples = np.zeros((numberOfTrainingSamples, 1))
    
    testSamples = np.zeros((numberOfTestSamples, numberOfFeatures))
    labelsTestSamples = np.zeros((numberOfTestSamples, 1))

    # print(trainingSamples.shape)
    # print(labelsTrainingSamples.shape)
    # print()
    # print(testSamples.shape)
    # print(labelsTestSamples.shape)

    # enum = locations
    # print(enum)

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
            if  twentyPercentTestData == 4:
            #if  dataPoint == -1 or (dataPoint != 0 and np.mod(dataPoint, 5) == 4):
                if line.__contains__("Scanning"):
                    location = line.split(": ")[1].strip()
                    for i in range(0, len(locations)):
                        if locations[i].__eq__(location):
                            # print(i)
                            indexTestSample += 1

                            # print(trainingSamples)
                            # print(labelsTrainingSamples)
                            # print()
                            # print(testSamples)
                            # print(labelsTestSamples)

                            labelsTestSamples[indexTestSample] = i
                
                if line.__contains__("ResultLevel"):
                    resultLevel = line.split(": ")[1].split()
                    # print("currentBSSID", currentBSSID)
                    # print("resultLevel", resultLevel)
                    testSamples = changeMatrice(testSamples, indexTestSample, distinctBSSID, currentBSSID[0], resultLevel[0])
                    
            else:
                if line.__contains__("Scanning"):
                    location = line.split(": ")[1].strip()
                    for i in range(0, len(locations)):
                        if locations[i].__eq__(location):
                            indexTrainingSample += 1
                            labelsTrainingSamples[indexTrainingSample] = i

                if line.__contains__("ResultLevel"):
                    resultLevel = line.split(": ")[1].split()
                    # print("currentBSSID", currentBSSID)
                    # print("resultLevel", resultLevel)
                    trainingSamples = changeMatrice(trainingSamples, indexTrainingSample, distinctBSSID, currentBSSID[0], resultLevel[0])
                        
                #print(line.strip())
    
    return trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples

def changeMatrice(matrice, index, distinctBSSID, currentBSSID, resultLevel):
    for i in range(0, len(distinctBSSID)):
        # print()
        # print("distinct", distinctBSSID[i])
        # print("current", currentBSSID[0])
        # print(distinctBSSID[i] == currentBSSID[0])
        # print()

        if distinctBSSID[i] == currentBSSID:
            # print("result", resultLevel)
            matrice[index, i] = resultLevel
            # print(matrice)
            return matrice    

def calculationsSVM(trainingSamples, labelsTrainingSamples, testSamples):
    clf = svm.SVC()
    clf.fit(trainingSamples, labelsTrainingSamples)
    prediction = clf.predict(testSamples)

    return prediction
    # print("prediction: \n", prediction)

    
def accuracySVM(labelsTestSamples, predictionSVM, locations):
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
            if labelsTestSamples[i] == 1:
                if predictionSVM[i] == 0: shouldBeLivingRoomPredictsOffice += 1
                else: shouldBeLivingRoomPredictsKitchen += 1
            else: 
                if predictionSVM[i] == 1: shouldBeOfficePredictsLivingRoom += 1
                else: shouldBeOfficePredictsKitchen += 1

            # print("Predicted: ", locations[predictionSVM[i].astype(int)])
            # print("Korrekt: ", locations[labelsTestSamples[i].astype(int)[0]])
            # print()
    
    accuracy = correct/len(labelsTestSamples)
    print()
    print("********************************************************************************************")
    print("RESULT SUPPORT VECTOR MACHINE")
    print()
    print("Overall accuracy SVM is %2.2f percentage of %d tested data points." % (accuracy*100, len(labelsTestSamples)))
    print()
    print()
    print("Details for the wrong predictions")
    print()
    print("Wrong predicitions in total: ", wrong)
    print("Should be office but predicted kitchen %d corresponds to %1.3f percentage of wrongs." % (shouldBeOfficePredictsKitchen, shouldBeOfficePredictsKitchen/wrong*100))
    print("Should be office but predicted living room %d corresponds to %1.3f percentage of wrongs." % (shouldBeOfficePredictsLivingRoom, shouldBeOfficePredictsLivingRoom/wrong*100))
    print("Should be kitchen but predicted office %d corresponds to %1.3f percentage of wrongs." % (shouldBeKitchenPredictsOffice, shouldBeKitchenPredictsOffice/wrong*100))
    print("Should be kitchen but predicted living room %d corresponds to %1.3f percentage of wrongs." % (shouldBeKitchenPredictsLivingRoom, shouldBeKitchenPredictsLivingRoom/wrong*100))
    print("Should be living room but predicted office %d corresponds to %1.3f percentage of wrongs." % (shouldBeLivingRoomPredictsOffice, shouldBeLivingRoomPredictsOffice/wrong*100))
    print("Should be living room but predicted kitchen %d corresponds to %1.3f percentage of wrongs." % (shouldBeLivingRoomPredictsKitchen, shouldBeLivingRoomPredictsKitchen/wrong*100))
    print()
    print("********************************************************************************************")


if __name__ == '__main__':
    main()