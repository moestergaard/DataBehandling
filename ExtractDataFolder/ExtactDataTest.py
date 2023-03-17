import unittest
import numpy as np
from ExtractData import extractDistinctBSSIDAndNumberOfDataPoints, extractData

class TestExample1(unittest.TestCase):
   
    def testExtractDistinctBSSIDAndNumberOfDataPoints(self):
        filename = 'WifiData2303141637Modified.txt' 
        distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)

        self.assertEqual(62, len(distinctBSSID))
        self.assertEqual(7070, dataPoints)

    def testExtractData(self):
        filename = 'WifiData2303141637Modified.txt' 
        distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(filename)

        trainingSamples, labelsTrainingSamples, testSamples, labelsTestSamples = extractData(filename, distinctBSSID, dataPoints)
        indexOfDetLilleHus = 0
        for i in range(0, len(distinctBSSID)):
            if distinctBSSID[i] == "DetLilleHus":
                indexOfDetLilleHus = i

        self.assertEqual(0, labelsTestSamples[0])
        self.assertEqual(0, labelsTrainingSamples[1])
        self.assertEqual(-57, trainingSamples[0,indexOfDetLilleHus])
        self.assertEqual(-63, testSamples[1, indexOfDetLilleHus])



if __name__ == '__main__':
    unittest.main()
