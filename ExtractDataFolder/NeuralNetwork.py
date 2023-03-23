import numpy as np

def calculationsNN(trainingSamples, labelsTrainingSamples, testSamples):
    one_hot_labels = np.zeros((len(labelsTrainingSamples), 3))

    for i in range(len(labelsTrainingSamples)):
        one_hot_labels[i, labelsTrainingSamples[i].astype(int)] = 1

    # print(one_hot_labels)

    instances = trainingSamples.shape[0]
    attributes = trainingSamples.shape[1]
    hidden_nodes = 4
    output_labels = 3

    # wh = np.random.randint(-1, 1, size=(attributes,hidden_nodes), dtype=np.float64)

    np.random.seed(42)

    wh = np.random.randn(attributes,hidden_nodes)
    bh = np.random.randn(hidden_nodes)

    # wo = np.random.randint(-1,1, size=(hidden_nodes,output_labels), dtype = np.float64)
    wo = np.random.randn(hidden_nodes,output_labels)
    bo = np.random.randn(output_labels)
    lr = 10e-4

    error_cost = []
    wh_list = []
    bh_list = []
    wo_list = []
    bo_list = []

    for epoch in range(500):
    ############# feedforward

        # Phase 1
        zh = np.dot(trainingSamples, wh) + bh
        ah = sigmoid(zh)
        # ah = zh

        # Phase 2
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)
        # ao = zo

    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = ao - one_hot_labels
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo

    ########## Phases 2

        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        # print()
        # print()
        # print(zh.shape)
        # print()
        # print()
        # dah_dzh = np.ones((zh.shape[0], zh.shape[1]))
        # print(dah_dzh.shape)
        # print(dah_dzh)
        # print()
        # print()
        # print()
        # dah_dzh = 1
        dzh_dwh = trainingSamples
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        wh -= lr * dcost_wh
        bh -= lr * dcost_bh.sum(axis=0)

        wo -= lr * dcost_wo
        bo -= lr * dcost_bo.sum(axis=0)
        # print('bos ændring: ', lr * dcost_bo.sum(axis=0))
        # print('den nye bo: ', bo)
        
        if epoch % 200 == 0:
            # print('ao: ', ao)
            
            # loss = -(np.sum(one_hot_labels * zo - one_hot_labels * np.sum(zo)))
            loss = np.sum(-one_hot_labels * np.log(ao))
            # print('Loss function value: ', loss)
            error_cost.append(loss)
            wh_list.append(wh.copy())
            bh_list.append(bh.copy())
            wo_list.append(wo.copy())
            bo_list.append(bo.copy())
            
            # print('bo hver gang der bliver appendes: ', bo)
        
        

    # print('one_hot_labels: ', one_hot_labels)
    # print('ao: ', ao)
    i = np.argmin(error_cost)
    
    # print('i: ', i)
    # print('Error Cost: ', error_cost[i])
    # print('Error Cost: ', error_cost[len(error_cost)-1])
    # # print('wh_i: ', wh_list[i])
    # # print('wh:', wh)
    # print('bh_i: ', bh_list[i])
    # print('bh:', bh)
    # # print('wo_i: ', wo_list[i])
    # # print('wo:', wo)
    # print('bo_i: ', bo_list[i])
    # print('bo:', bo)
    # print('bo listen: ', bo_list)
    
    #return wh, bh, wo, bo, error_cost, error_cost[len(error_cost)-1]
    return wh_list[i], bh_list[i], wo_list[i], bo_list[i], error_cost, error_cost[i]

def accuracyNN(testSamples, labelsTestSamples, wh, bh, wo, bo, error_cost_list, error_cost):
    predictedLabels = []
    percentageSure = []

    zh = np.dot(testSamples, wh) + bh
    print("shape of test samples: ", testSamples.shape)
    print("shape of zh: ", wh.shape)
    print("shape of bh: ", bh.shape)
    print("shape of zh: ", zh.shape)
    ah = sigmoid(zh)

    z0 = np.dot(ah, wo) + bo
    ah = softmax(z0)

    for i in range(len(ah)):
        predictedLabels.append(np.argmax(ah[i]))
        # if predictedLabels[i] == labelsTestSamples[i]:
        percentageSure.append(np.max(ah[i]))

    print("\n", error_cost_list)

    printAccuracy(labelsTestSamples, predictedLabels, wh, bh, wo, bo, error_cost, percentageSure)

    # correct = 0

    # for i in range(len(labelsTestSamples)):
    #     if (labelsTestSamples[i] == predictedLabels[i]):
    #         correct += 1
        
    # print("*****************************************************************")
    # print('Error Cost: ', error_cost)
    # print('Accuracy: ', correct/len(labelsTestSamples))
    # print("*****************************************************************")

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

def printAccuracy(labelsTestSamples, predictedLabels, wh, bh, wo, bo, error_cost, percentageSure):
    correct = 0
    wrong = 0

    shouldBeKitchenPredictsLivingRoom = 0
    shouldBeKitchenPredictsOffice = 0
    shouldBeLivingRoomPredictsKitchen = 0
    shouldBeLivingRoomPredictsOffice = 0
    shouldBeOfficePredictsKitchen = 0
    shouldBeOfficePredictsLivingRoom = 0

    for i in range(0, len(labelsTestSamples)):
        if labelsTestSamples[i] == predictedLabels[i]:
            correct += 1
        else:
            wrong += 1
            if labelsTestSamples[i] == 2:
                if predictedLabels[i] == 0: shouldBeKitchenPredictsOffice += 1
                else: shouldBeKitchenPredictsLivingRoom += 1
            elif labelsTestSamples[i] == 1:
                if predictedLabels[i] == 0: shouldBeLivingRoomPredictsOffice += 1
                else: shouldBeLivingRoomPredictsKitchen += 1
            elif labelsTestSamples[i] == 0:
                if predictedLabels[i] == 1: shouldBeOfficePredictsLivingRoom += 1
                else: shouldBeOfficePredictsKitchen += 1
                

    accuracy = correct/len(labelsTestSamples)
    print()
    print()
    print("RESULT NEURAL NETWORK")
    print()
    print("bh: \n", bh)
    print()
    print("wh: \n", wh)
    print()
    print("bo: \n", bo)
    print()
    print("wo: \n", wo)
    print()
    print("error cost: ", error_cost)
    print()
    print("The 25-fractile of the percentage sure is %2.2f" % (np.percentile(percentageSure, 25)*100))
    print("The 50-fractile of the percentage sure is %2.2f" % (np.percentile(percentageSure, 50)*100))
    print("The 75-fractile of the percentage sure is %2.2f" % (np.percentile(percentageSure, 75)*100))
    print()
    print("Overall accuracy NN is %2.2f percentage of %d tested data points." % (accuracy*100, len(labelsTestSamples)))
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