import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster import hierarchy


def plot(link, x, y):
    plt.figure()
    plt.title(link)
    plt.ylabel("Distance")
    z = hierarchy.linkage(x, link.lower())
    hierarchy.dendrogram(z, labels=y)

    plt.show()


if __name__ == "__main__":
    np.random.seed(1)

    X = np.zeros((4, 1))
    X[0][0] = 0
    X[1][0] = 5
    X[2][0] = 9
    X[3][0] = 12

    labels = ["A", "B", "C", "D"]

    plot("Single", X, labels)
    plot("Complete", X, labels)
    plot("Average", X, labels)
