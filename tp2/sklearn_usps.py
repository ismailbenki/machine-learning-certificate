import numpy as np
from matplotlib import pyplot

from sklearn.linear_model.perceptron import Perceptron
from sklearn.metrics import accuracy_score

import usps
import perceptron # Pour la fonction two_classes

data, labels           = usps.load_train()
data_test, labels_test = usps.load_test()

for k in range(10):
    labels_k = perceptron.two_classes(labels, k)

    net = Perceptron()
    net.fit(data, labels_k)
    output_train = net.predict(data)
    output_test  = net.predict(data_test)
    print(k)
    print("  Score (train)", accuracy_score(labels_k, output_train))
    labels_k_test = perceptron.two_classes(labels_test, k)
    print("  Score (test)", accuracy_score(labels_k_test, output_test))
