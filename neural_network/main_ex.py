from turtle import right
import numpy as np

from ann_ex import ANN

from utils_ex import plot_decision_boundary, print_dataset, load_dataset


def test_ann():
    X, Y = load_dataset(500)

    # TODO: inizializza n_x, n_h, n_y
    n_x = 2  # numero di feature
    n_h = 4  # numero di neuroni nell'hidden layer
    n_y = 1  # numero di neuroni in output

    print_dataset(X, Y.T)

    # TODO istanzia un oggetto ANN e chiama il metodo fit
    model = ANN(n_x, n_h, n_y)
    model.fit(X, Y)

    plot_decision_boundary(model, X, Y)
    
    Y_cap = model.predict(X)
    print(Y_cap)

    # # TODO: implementa la funzione accuracy
    acc = accuracy(Y,Y_cap)
    print(f'Accuracy: {acc:.2f}')



def accuracy(y_true, y_pred):
    # TODO: implementa un metodo che calcoli la funzione di costo
    right = np.sum(np.array([y_true == y_pred]).astype(int))
    return (right / y_true.shape[1]) * 100


if __name__ == '__main__':
   test_ann()
   exit(0)