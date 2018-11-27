import numpy as np
from matplotlib import pyplot

import perceptron
import usps

def accuracy(truth, output):
    n = truth.shape[0]
    return (truth == output).sum() / n

def accuracy(truth, output):
    n = truth.shape[0]
    accur = 0.
    for i in range(n):
        if truth[i] == output[i]:
            accur += 1
    return accur / n

data, labels           = usps.load_train()
data_test, labels_test = usps.load_test()

for k in range(10):
    labels_k = perceptron.two_classes(labels, k)
    weights, errors = perceptron.train(data, labels_k, with_errors=True)

    print(k)
    output = np.array([ perceptron.predict(weights, x) for x in data ])
    print("  Score (train)", accuracy(labels_k, output))
    output = np.array([ perceptron.predict(weights, x) for x in data_test ])
    print("  Score (test)", accuracy(perceptron.two_classes(labels_test, k), output))

    pyplot.clf()
    pyplot.imshow(weights[:-1].reshape((16, 16)), cmap=pyplot.gray())
    pyplot.colorbar()
    pyplot.savefig("usps_" + str(k) + "-weights.png")

    pyplot.plot(errors)
    pyplot.savefig("usps_" + str(k) + "-errors.png")
