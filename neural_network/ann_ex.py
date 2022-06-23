from re import A
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt

import activation_ex


class ANN:

    def __init__(self, n_x, n_h, n_y):
        np.random.seed(2)

        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

        self.cost = []

        # TODO inizializza i pesi utilizzando la regola di Glorot
        """
        fanin = neuroni in input
        fanout = neuroni in output
        fanavg = (fanin + fanout) / 2
        1/fanavg
        """
        self.W1 = np.random.randn(self.n_h, self.n_x) * np.sqrt(1/((self.n_x + self.n_h)/2) )
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * np.sqrt(1/((self.n_y + self.n_h)/2))
        self.b2 = np.zeros((n_y, 1))
        # print(self.W1.shape)
        # print(self.b1.shape)
        # print(self.W2.shape)
        # print(self.b2.shape)

    def forward_propagation(self, X):

        Z1 = np.dot(self.W1, X) + self.b1
        A1 = activation_ex.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = activation_ex.sigmoid(Z2)

        return {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}

    def compute_cost(self, Y, A2):
        # TODO: implementa la funzione di costo
        L = -1 / len(Y) * (np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)))
        return L


    def back_propagation(self, X, Y, output, lambd):

        m = X.shape[1]
        div = 1 / m

        A1 = output['A1']
        A2 = output['A2']

        dZ2 = A2 - Y
        dW2 = div * np.dot(dZ2, A1.T)
        db2 = div * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), activation_ex.tanh_der(A1))
        dW1 = div * np.dot(dZ1, X.T)
        db1 = div * np.sum(dZ1, axis=1, keepdims=True)

        assert (dZ2.shape == (self.n_y, m))
        assert (dW2.shape == (self.n_y, self.n_h))
        assert (db2.shape == (self.n_y, 1))
        assert (dZ1.shape == (self.n_h, m))
        assert (dW1.shape == (self.n_h, self.n_x))
        assert (db1.shape == (self.n_h, 1))

        # TODO: Aggiorna i parametri applicando il metodo del gradiente
        self.W1 = self.W1 - lambd * dW1
        self.b1 = self.b1 - lambd * db1
        self.W2 = self.W2 - lambd * dW2
        self.b2 = self.b2 - lambd * db2

    def fit(self, X, Y, n_iter=10000, lambd=0.4):

        for i in range(0, n_iter):
            # TODO: implementa la fase di training utilizzando
            #  i metodi forward_propagation, back_propagation
            #  e compute_cost
            output = self.forward_propagation(X)
            self.back_propagation(X, Y, output, lambd)
            self.cost.append(self.compute_cost(Y, output['A2']))

        plt.figure(figsize=(15,5))
        plt.plot(self.cost)
        plt.savefig('./images/Costo.png')

    def predict(self, X):
        # TODO: implementa il metodo predict
        A2 = self.forward_propagation(X)["A2"]
        A2[A2 > .5] = 1
        A2[A2 <= .5] = 0
        return A2
