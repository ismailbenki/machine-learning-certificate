import numpy as np
from matplotlib import pyplot

import perceptron

n = 10
d = 2

data0 = np.random.randn(n, d) + 3
data1 = np.random.randn(n, d) - 3
data  = np.concatenate([data0, data1])

labels = np.concatenate([np.zeros(n), np.ones(n)]) # Deux classes étiquettées 0 et 1
labels = perceptron.two_classes(labels, 0)         # Deux classes étiquettées -1 et 1

weights, errors = perceptron.train(data, labels, with_errors=True)
print(weights)

for i in range(data.shape[0]):
    print(i, labels[i], perceptron.predict(weights, data[i]), data[i])

pyplot.scatter(data0[:,0], data0[:,1], marker="x", color="r", s=100)
pyplot.scatter(data1[:,0], data1[:,1], marker="*", color="b", s=100)
x0 = 0
y0 = -weights[2]/weights[1]
x1 = -weights[2]/weights[0]
y1 = 0
a = (y1 - y0) / (x1 - x0)
b = y0
pyplot.plot([-10, +10], [-10 * a + b, +10 * a + b], color="g")
pyplot.xlim(-6, 6)
pyplot.ylim(-6, 6)
pyplot.show()

pyplot.plot(errors)
pyplot.show()
