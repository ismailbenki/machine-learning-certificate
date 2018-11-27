import numpy as np
import matplotlib.pyplot as plt

def h(x):
    if x<0:
        return -1
    else:
        return 1

def F(w,x):
    return np.dot(w,x)

def cost(w,x,y):
    return .5*(y - F(w,x))**2

def grad_cost(w,x,y):
    return (y - F(w,x)) * np.array(x)

def two_classes(labels, label_of_interest):
    n = labels.shape[0]
    ones = np.ones(n)
    minus_ones = -np.ones(n)
    return np.select([labels == label_of_interest,
                      labels != label_of_interest],
                     [ones, minus_ones])

def init_biais(data):
    n,d = data.shape
    ones = np.ones((n,1))
    data_biais = np.concatenate([ones,data],axis=1)
    return data_biais

def update_weights(weights,data,labels,alpha):
    err = 0
    n = len(labels)
    for i in range(n):
        x,y = data[i],labels[i]
        err += cost(weights,x,y)
        if (h(y*F(weights,x)) == 1):
            weights += alpha * grad_cost(weights,x,y)
    return weights,err/n

def train(data0,labels,alpha=1e-3,eps=1e-9, max_it=1e2,show_err=False):
    converged = False
    data = init_biais(data0)
    weights = np.zeros(len(data[0]))
    it_count = 0
    glob_err = []
    err_old = 0

    while not converged and it_count < max_it :
        weights,err = update_weights(weights,data,labels,alpha)
        converged = abs(err - err_old) < eps
        err_old = err
        glob_err.append(err)
        it_count += 1

    if show_err:
        return weights,glob_err
    else:
        return weights

def predict(data0,weights):
    data = np.insert(data0,0,1)
    return h(F(weights,data))

def plot_perceptron(data,labels):
    weights = train(data,labels)
    x0 = 0
    y0 = -weights[0]/weights[1]
    x1 = -weights[0]/weights[2]
    y1 = 0
    a = (y1 - y0) / (x1 - x0)
    b = y0
    plt.plot([-10, +10], [-10 * a + b, +10 * a + b], color="g")

def plot_multiperceptron(data,labels):
    classes = np.unique(labels)
    if len(classes)>2:
        for c in classes:
            plot_perceptron(data, two_classes(labels,c))
    else:
        plot_perceptron(data,labels)
