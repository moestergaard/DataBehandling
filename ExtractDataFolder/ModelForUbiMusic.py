from ExtractData import extractDistinctBSSIDAndNumberOfDataPoints, extractData, randomSplitSamplesAndLabels
from NeuralNetwork import bestModelNN

def main():
    locations = ["Kontor", "Stue", "Køkken"]
    dataSet = "Data/WifiData230424_17-21.txt"
    
    partOfData = 1/9
    
    distinctBSSID, dataPoints = extractDistinctBSSIDAndNumberOfDataPoints(locations, dataSet)
    samples, labels = extractData(locations, dataSet, distinctBSSID, dataPoints)
    
    trainingSamplesOverall, _, trainingLabelsOverall, _ = randomSplitSamplesAndLabels(samples, labels, partOfData)
    
    wh, bh, wo, _, _, accuracy = bestModelNN(trainingSamplesOverall, trainingLabelsOverall, bias = False, activationFunction = 'identity', numberOfClasses=len(locations))
    
    result = printMatrice(wh)
    print(result)
    print()
    result = printMatrice(wo)
    print(result)
    print()
    print(len(bh))
    print()
    print(accuracy)
    
def printMatrice(matrice):
    return ''.join([f'{{{", ".join([f"{col:.15f}" for col in row])}}}' for row in matrice])

if __name__ == '__main__':
    main()