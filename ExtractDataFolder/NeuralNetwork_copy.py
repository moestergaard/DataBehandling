import numpy as np

def calculationsNN(trainingSamples, labelsTrainingSamples, testSamples):
    one_hot_labels = np.zeros((len(labelsTrainingSamples), 3))

    for i in range(len(labelsTrainingSamples)):
        one_hot_labels[i, labelsTrainingSamples[i].astype(int)] = 1

    print(one_hot_labels)

    instances = trainingSamples.shape[0]
    attributes = trainingSamples.shape[1]
    hidden_nodes = 15
    output_labels = 3

    # wh = np.random.randint(-1, 1, size=(attributes,hidden_nodes), dtype=np.float64)
    wh = np.random.randn(attributes,hidden_nodes)
    bh = np.random.randn(hidden_nodes)

    # wo = np.random.randint(-1,1, size=(hidden_nodes,output_labels), dtype = np.float64)
    wo = np.random.randn(hidden_nodes,output_labels)
    bo = np.random.randn(output_labels)
    lr = 10e-4

    error_cost = []

    for epoch in range(5):
    ############# feedforward

        # Phase 1           Fra input lag til hidden lag
        zh = np.dot(trainingSamples, wh) + bh
        # ah = zh

        # # Phase 2         Fra hidden lag til output lag - bemærk, at softmax ikke beregnes, da det er medtaget i loss function
        zo = np.dot(zh, wo) + bo
        # ao = softmax(zo)

    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = softmax(zo) - one_hot_labels
        dzo_dwo = zh

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo

    ########## Phases 2

        dzo_dzh = wo
        dcost_dzh = np.dot(dcost_dzo , dzo_dzh.T)
        # dah_dzh = 1
        
        dzh_dwh = trainingSamples
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh)

        dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        wh -= lr * dcost_wh
        bh -= lr * dcost_bh.sum(axis=0)

        #wo -= lr * dcost_wo
        #bo -= lr * dcost_bo.sum(axis=0)

        if epoch % 200 == 0:
            # print('ao: ', ao)
            loss = -(np.sum(one_hot_labels * zo - one_hot_labels * np.sum(zo)))
            
            # loss = np.sum(-one_hot_labels * np.log(ao))
            print('Loss function value: ', loss)
            error_cost.append(loss)

    print('one_hot_labels: ', one_hot_labels)
    print('ao: ', ao)
    return 0

def accuracyNN(labelsTestSamples, predictionNN):
    tmp = 0

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