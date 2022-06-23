import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y):
    # Recupera il minimo e il massino da X e y
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Genera una griglia di punti
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Fa una predizione (0 o 1) per ogni punto nella griglia
    # Traspongo per avere sulle righe le features e sulle colonne le istanze
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)

    # Reshape altrimenti avrei un vettore riga invece di una matrice
    Z = Z.reshape(xx.shape)

    # Uso la funzione countourf per vedere come varia la predizione nei punti della griglia
    plt.figure(figsize=(15,5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.savefig('./images/decision_boundary')


def load_dataset(n_samples):
    X, Y = sklearn.datasets.make_circles(n_samples=n_samples, shuffle=True, noise=0.1, random_state=0, factor=0.4)
    X = X.T
    Y = Y.reshape(1, -1)
    return X, Y


def print_dataset(X, Y, img_name='data'):
    plt.figure(figsize=(15,5))
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.savefig(f'./images/{img_name}')
