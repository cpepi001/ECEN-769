import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

if __name__ == "__main__":
    # linear data
    X = np.array([2, 2, 3, 3, 4, 4, 5, 5])
    y = np.array([3, 5, 2, 4, 3, 5, 2, 4])

    # shaping data for training the model
    training_X = np.vstack((X, y)).T
    training_y = [1, 1, 0, 1, 0, 1, 0, 0]

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1)
    clf.fit(training_X, training_y)

    plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    # CART
    plt.axhline(y=3.5, color="r", linestyle="-")

    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
