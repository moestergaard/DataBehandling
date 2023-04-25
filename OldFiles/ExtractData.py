import numpy as np

#
# Makes an array consisting of the distinct BSSID.
#
def extractDistinctBSSIDAndNumberOfDataPoints(filename, distinctBSSIDTraining = []):
    # print("distinctbssidtraining: ", len(distinctBSSIDTraining))
    if(distinctBSSIDTraining != []):
        distinctBSSID = [0] * len(distinctBSSIDTraining)
    else: distinctBSSID = []
    # print("distinctbssid: ", len(distinctBSSID))
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
                    if(distinctBSSIDTraining != []):
                        for i in range(0, len(distinctBSSIDTraining)):
                            if bssid.__eq__(distinctBSSIDTraining[i]):
                                distinctBSSID[i] = bssid
                    else: distinctBSSID.append(bssid)
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
    
    # print("samples: ", samples.shape)
    # print("labels: ", labels.shape)
        
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
                # print('samples: ', samples)
                samples = changeMatrice(samples, index, distinctBSSID, currentBSSID[0], resultLevel[0])

    return samples, labels

def changeMatrice(matrice, index, distinctBSSID, currentBSSID, resultLevel):
    # if(matrice.any() == None):
    #     print('**************')
    # print('samples: ', matrice)
    # print('index: ', index)
    # print('distinctBSSID: ', distinctBSSID)
    # print('currentBSSID: ', currentBSSID)
    # print('resultLevel: ', resultLevel)
    for i in range(0, len(distinctBSSID)):
        if distinctBSSID[i] == currentBSSID:
            matrice[index, i] = resultLevel
            return matrice  
    return matrice  


def changeDataFile(filename):
    # bssid = ""
    # resultlevel = ""

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("Scanning"):
                print("**************")
                print()
            if line.__contains__("SignalLevel"):
                ()
            # if line.__contains__("BSSID"):
            #     bssid = line.strip()
            # elif line.__contains__("ResultLevel"):
            #     resultlevel = line.strip()
            # elif line.__contains__("Frequency"):
            #     bssid += "+"
            #     bssid += line.split(": ")[1].split()[0]
            #     print(bssid)
            #     print(resultlevel)
            #     print(line.strip())
            else: print(line.strip())