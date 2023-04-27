import numpy as np
from MatrixManipulation import changeMatrice

def extractDistinctBSSIDAndNumberOfDataPoints(filename, locations, distinctBSSIDTraining = []):
    if(distinctBSSIDTraining != []):
        distinctBSSID = [0] * len(distinctBSSIDTraining)
    else: distinctBSSID = []
    dataPoints = 0
    dataPointIncluded = False

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("BSSID"):
                if dataPointIncluded:
                    bssid = line.split(": ")[1].strip()
                    alreadyIncludedInDistinctBSSID = distinctBSSID.__contains__(bssid)
                    if not alreadyIncludedInDistinctBSSID:
                        if(distinctBSSIDTraining != []):
                            for i in range(0, len(distinctBSSIDTraining)):
                                if bssid.__eq__(distinctBSSIDTraining[i]):
                                    distinctBSSID[i] = bssid
                        else: distinctBSSID.append(bssid)
            if line.__contains__("Scanning"):
                dataPointIncluded = False
                for i in range(0, len(locations)):
                    if line.__contains__(locations[i]):    
                        dataPoints += 1
                        dataPointIncluded = True

    return distinctBSSID, dataPoints

def extractData(filename, distinctBSSID, numberOfSamples, locations):

    samples = np.zeros((numberOfSamples, len(distinctBSSID)))
    labels = np.zeros((numberOfSamples, ))
    
    index = 0

    currentBSSID = ""
    dataPointIncluded = False

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("********"): 
                index += 1
            if line.__contains__("BSSID"):
                if dataPointIncluded:
                    currentBSSID = line.split(": ")[1].split()
            if line.__contains__("Scanning"):
                dataPointIncluded = False
                location = line.split(": ")[1].strip()
                for i in range(0, len(locations)):
                    if locations[i].__eq__(location): 
                        dataPointIncluded = True
                        labels[index] = i
                if not dataPointIncluded:
                    index -= 1
            if line.__contains__("ResultLevel"):
                if dataPointIncluded:
                    resultLevel = line.split(": ")[1].split()
                    samples = changeMatrice(samples, index, distinctBSSID, currentBSSID[0], resultLevel[0])

    return samples, labels


def extractDataFromMultipleFiles(filenameTests, locations, distinctBSSID):
    listOfTestSamples = []
    listOfTestLabels = []
    for filenameTest in filenameTests:
        _ , dataPointsTest = extractDistinctBSSIDAndNumberOfDataPoints(filenameTest, locations, distinctBSSID)
        testSamples, testLabels = extractData(filenameTest, distinctBSSID, dataPointsTest, locations)
        
        listOfTestSamples.append(testSamples)
        listOfTestLabels.append(testLabels)
        
    allTestSamples = np.concatenate(listOfTestSamples)
    allTestLabels = np.concatenate(listOfTestLabels)
    
    return allTestSamples, allTestLabels










# #
# # Makes the following matrices
# #   For training: m is the number of training samples, and n is the number of features (distinct BSSID)
# #   A mxn matrice
# #   A mx1 matrice:  The labels corresponding to training samples.
# #  
# #   For testing: k is the number of test samples, and n is the same as before.
# #   A kxn matrice
# #   A kx1 matrice:  The labels corresponding to test samples.
# #
# def extractData(filename, distinctBSSID, samples, locations, ratio):

#     numberOfFeatures = len(distinctBSSID)
#     numberOfTestSamples = np.ceil(samples/5).astype(int)
#     numberOfTrainingSamples = samples - numberOfTestSamples

#     trainingSamples = np.zeros((numberOfTrainingSamples, numberOfFeatures))
#     labelsTrainingSamples = np.zeros((numberOfTrainingSamples, ))
    
#     testSamples = np.zeros((numberOfTestSamples, numberOfFeatures))
#     labelsTestSamples = np.zeros((numberOfTestSamples, ))

#     twentyPercentTestData = 4
#     indexTestSample = -1
#     indexTrainingSample = -1
    
#     dataPointIncluded = False

#     currentBSSID = ""
    

#     with open(filename) as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 break
#             if line.__contains__("********"):
#                 twentyPercentTestData = np.mod(twentyPercentTestData + 1, 5)
#             if line.__contains__("BSSID"):
#                 if dataPointIncluded:
#                     currentBSSID = line.split(": ")[1].split()
#             if line.__contains__("Scanning"):
#                 dataPointIncluded = False
#                 location = line.split(": ")[1].strip()
#                 for i in range(0, len(locations)):
#                     if locations[i].__eq__(location):
#                         if twentyPercentTestData == 4:
#                             indexTestSample += 1
#                             labelsTestSamples[indexTestSample] = i
#                             dataPointIncluded = True
#                         else:
#                             indexTrainingSample += 1
#                             labelsTrainingSamples[indexTrainingSample] = i
#                             dataPointIncluded = True
#                 if not dataPointIncluded:
#                     if twentyPercentTestData == 0: twentyPercentTestData = 4
#                     else: twentyPercentTestData -= 1
#             if line.__contains__("ResultLevel"):
#                 if dataPointIncluded:
#                     resultLevel = line.split(": ")[1].split()
#                     if twentyPercentTestData == 4:
#                         testSamples = changeMatrice(testSamples, indexTestSample, distinctBSSID, currentBSSID[0], resultLevel[0])
#                     else: trainingSamples = changeMatrice(trainingSamples, indexTrainingSample, distinctBSSID, currentBSSID[0], resultLevel[0])

#     tmpRatio = 0
#     tmpTrainingSamples = trainingSamples
    
#     return trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples

# def parseWifiData(filename, locations):
#     bssid_indices = {}
#     scanning_indices = {}
#     num_data_points = 0
#     num_scans = 0
#     num_bssid = 0
#     distinct_bssid = set()

#     # read the input file
#     with open(filename) as f:
#         lines = f.readlines()

#     # process each line in the input file
#     for line in lines:
#         # check if the line starts with "Scanning:"
#         if line.startswith("Scanning:"):
#             # extract the location name
#             location = line.strip().split(": ")[1]
#             # check if the location is in the list of known locations
#             if location in locations:
#                 num_data_points += 1
#                 # map the location name to an index
#                 # if location not in scanning_indices:
#                 #     # scanning_indices[location] = num_scans
#                 #     num_scans += 1
    
#         # check if the line starts with "BSSID:"
#         elif line.startswith("BSSID:"):
#             # extract the BSSID value
#             bssid = line.strip().split(": ")[1]
#             # map the BSSID value to an index
#             if bssid not in bssid_indices:
#                 bssid_indices[bssid] = num_bssid
#                 num_bssid += 1
#                 distinct_bssid.add(bssid)
    
#     # create the samples matrix
#     samples = np.zeros((num_data_points, num_bssid))

#     # create the labels matrix
#     labels = np.zeros((num_data_points, ), dtype=np.int32)
    
#     # create the index
#     index_data_points = -1
    
#     # boolean value indicating whether the current data point is included
#     included_data_point = False
    
#     # current bssid
#     current_bssid = ""

#     # process each line in the input file
#     for line in lines:
#         # check if the line starts with "Scanning:"
#         if line.startswith("Scanning:"):
#             # extract the location name
#             location = line.strip().split(": ")[1]
#             # check if the location is in the list of known locations
#             # if location in scanning_indices:
#             #     # get the index of the current location
#             #     location_index = scanning_indices[location]
#             #     # set the current label
#             #     labels[location_index, 0] = locations.index(location)
#             if location in locations:
#                 index_data_points += 1
#                 labels[index_data_points, ] = locations.index(location)
                
#                 included_data_point = True
#                 #samples[index_data_points, :] = -100
        
#         # check if the line starts with "BSSID:"
#         elif line.startswith("BSSID:"):
#             if included_data_point:
#                 # extract the BSSID value
#                 bssid = line.strip().split(": ")[1]
#                 # map the BSSID value to an index
#                 bssid_index = bssid_indices[bssid]
#                 current_bssid = bssid
        
#         # check if the line starts with "ResultLevel:"
#         elif line.startswith("ResultLevel:"):
#             if included_data_point:
#                 # extract the ResultLevel value
#                 result_level = int(line.strip().split(": ")[1])
#                 # get the index of the current location
#                 location_index = scanning_indices.get(location, None)
#                 # get the index of the current BSSID
#                 bssid_index = bssid_indices.get(current_bssid, None)
#                 # update the samples matrix
#                 # if location_index is not None and bssid_index is not None:
#                 #     samples[location_index, bssid_index] = result_level
#                 # update the samples matrix
#                 if bssid_index is not None:
#                     samples[index_data_points, bssid_index] = result_level
    
#     # listBssid = list(distinct_bssid)
#     listDistinctBSSID = list(bssid_indices.keys())
    
#     return samples, labels, listDistinctBSSID



