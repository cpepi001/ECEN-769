import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def plot_2(pred, centroids):
    c_0 = pred == 0
    c_1 = pred == 1

    x1 = data[:, 0]
    x2 = data[:, 1]

    plt.xlim(0, 5)
    plt.ylim(0, 4)
    plt.scatter(x1[c_0], x2[c_0], c="red")
    plt.scatter(x1[c_1], x2[c_1], c="green")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="blue")
    plt.show()


def plot_3(pred, centroids):
    c_0 = pred == 0
    c_1 = pred == 1
    c_2 = pred == 2

    x1 = data[:, 0]
    x2 = data[:, 1]

    plt.xlim(0, 5)
    plt.ylim(0, 4)
    plt.scatter(x1[c_0], x2[c_0], c="red")
    plt.scatter(x1[c_1], x2[c_1], c="green")
    plt.scatter(x1[c_2], x2[c_2], c="maroon")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="blue")
    plt.show()


def plot_4(pred, centroids):
    c_0 = pred == 0
    c_1 = pred == 1
    c_2 = pred == 2
    c_3 = pred == 3

    x1 = data[:, 0]
    x2 = data[:, 1]

    plt.xlim(0, 5)
    plt.ylim(0, 4)
    plt.scatter(x1[c_0], x2[c_0], c="red")
    plt.scatter(x1[c_1], x2[c_1], c="green")
    plt.scatter(x1[c_2], x2[c_2], c="maroon")
    plt.scatter(x1[c_3], x2[c_3], c="orange")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="blue")
    plt.show()


if __name__ == "__main__":
    data = np.array([[1, 1], [2, 1], [3, 1], [4, 1],
                     [1, 3], [2, 3], [3, 3], [4, 3]])

    k_2 = [0, 7]
    for r in k_2:
        k_means = KMeans(n_clusters=2, random_state=r).fit(data)
        plot_2(k_means.predict(data), k_means.cluster_centers_)

    k_4 = [30, 87]
    for r in k_4:
        k_means = KMeans(n_clusters=4, random_state=r).fit(data)
        plot_4(k_means.predict(data), k_means.cluster_centers_)

    k_3 = [2, 4]
    for r in k_3:
        k_means = KMeans(n_clusters=3, random_state=r).fit(data)
        plot_4(k_means.predict(data), k_means.cluster_centers_)
