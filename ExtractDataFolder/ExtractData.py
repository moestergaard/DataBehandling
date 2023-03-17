# from enum import Enum
import numpy as np

# class Location(Enum):
#     Kontor = 1
#     Stue = 2
#     Køkken = 3

def main():
    filename = 'WifiData2303141637Modified.txt' 
    #changeDataFile(filename)
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)
    print(distinctBSSID)
    print(len(distinctBSSID))
    print(dataPoints)
    trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints)

    #print(trainingSamples)
    print(labelsTrainingSamples)
    #print()
    #print(testSamples)
    #print(labelsTestSamples)


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
def extractData(filename, distinctBSSID, samples):
    locations = ["Kontor", "Stue", "Køkken"]

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
    

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"):
                twentyPercentTestData = np.mod(twentyPercentTestData + 1, 5)
            else:
                if  twentyPercentTestData == 4:
                #if  dataPoint == -1 or (dataPoint != 0 and np.mod(dataPoint, 5) == 4):
                    if line.__contains__("Scanning"):
                        location = line.split(": ")[1].strip()
                        for i in range(0, len(locations)):
                            if locations[i].__eq__(location):
                                # print(i)
                                indexTestSample += 1
                                labelsTestSamples[indexTestSample] = i
                                break
                        
                else:
                    if line.__contains__("Scanning"):
                        location = line.split(": ")[1].strip()
                        for i in range(0, len(locations)):
                            if locations[i].__eq__(location):
                                indexTrainingSample += 1
                                labelsTrainingSamples[indexTrainingSample] = i
                                break
                        
                #print(line.strip())
    
    return trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples

if __name__ == '__main__':
    main()