import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def gaussian_samples(n,N=2,scale=2,show=False):

    if N>4 or N<2:
        print("N must be in [2,4]")
        return False

    V = [[1,1],[-1,1],[1,-1],[-1,-1]]
    colors = ['red','green','yellow','pink']

    data=np.random.randn(n,2) + np.array(V[0]) * scale
    labels=np.zeros(n)
    if show:
        plt.scatter(data[:,0],data[:,1],color=colors[0], label='classe 0')

    for i in np.arange(1,N):
        data1 = np.random.randn(n,2) + np.array(V[i]) * scale
        labels1 = np.ones(n) * i
        data = np.concatenate([data,np.array(data1)])
        labels = np.concatenate([labels,labels1])
        if show:
            plt.scatter(data1[:,0],data1[:,1],color=colors[i], label='classe '+str(i))

    return data,labels

def display_classifier(clf,X,y,weights='uniform'):
    h = .02  # step size in the mesh

    cmap_light = ListedColormap(['#AAAAFF','#FFAAAA', '#AAFFAA',
                                 '#FFFFAA'])
    cmap_bold = ListedColormap(['blue','red', 'green', 'yellow'])


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    zz = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, zz, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()
