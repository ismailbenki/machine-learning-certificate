import numpy as np
from matplotlib import pyplot

def load_file(path):
    raw    = np.loadtxt(path)
    labels = raw[:, 0]  # Première colonne: chiffre
    data   = raw[:, 1:] # Deuxième colonne: pixels de l'image
    return data, labels

def load_train():
    return load_file("./zip.train.gz")

def load_test():
    return load_file("./zip.test.gz")

def display(img):
    img = img.reshape(16, 16)              # Mise sous forme matricielle de l'image
    pyplot.imshow(img, cmap=pyplot.gray()) # Affichage en niveau de gris
    pyplot.show()

if __name__ == "__main__":
    train_data, train_labels = load_train()
    print(train_data.shape, train_labels.shape)

    test_data, test_labels = load_test()
    print(test_data.shape, test_labels.shape)

    display(train_data[0])
