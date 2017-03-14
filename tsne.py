from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


# Scale and visualize the embedding vectors
def __plot_embedding__(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1((y[i] % 11.)/10),
                 fontdict={'weight': 'bold', 'size': 9})


def plot_original(X, y):
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    X_tsne = tsne.fit_transform(X)
    __plot_embedding__(X_tsne, y, "t-SNE embedding of the digits")
    plt.show()