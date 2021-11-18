import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot(X, Y, y, name_1, name_2):
    plt.xlabel("%s principal component" % name_1)
    plt.ylabel("%s principal component" % name_2)

    low = y <= 2
    medium = (y > 2) & (y <= 8)
    high = y > 8

    plt.scatter(X[low], Y[low], c="blue", s=5, marker="o", label="Low")
    plt.scatter(X[medium], Y[medium], c="green", s=5, marker="o", label="Medium")
    plt.scatter(X[high], Y[high], c="red", s=5, marker="o", label="High")

    plt.legend()
    plt.grid()
    plt.show()


def plot_2(X, Y, y):
    plt.xlabel("Fe")
    plt.ylabel("Si")

    low = y <= 2
    medium = (y > 2) & (y <= 8)
    high = y > 8

    plt.scatter(X[low], Y[low], c="blue", s=5, marker="o", label="Low")
    plt.scatter(X[medium], Y[medium], c="green", s=5, marker="o", label="Medium")
    plt.scatter(X[high], Y[high], c="red", s=5, marker="o", label="High")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Read the spreadsheet
    df = pd.read_csv("Soft_Magnetic_Alloy_Dataset.csv")

    data = pd.DataFrame(df[df.columns[0:26]])
    data["Coercivity (A/m)"] = df[df.columns[33]]

    # Discard all features (columns) that do not have at least 5% nonzero values
    for c in data.columns:
        if c == "Coercivity (A/m)":
            break

        if data[c][data[c] != 0].count() <= 0.05 * data.shape[0]:
            data = data.drop(c, axis=1)

    # Discard all entries (rows) that do not have a recorded coercivity value
    data = data.dropna(subset=["Coercivity (A/m)"])

    print(data.columns)

    # Split data
    X = data.iloc[:, 0: len(data.columns) - 1]
    y = data.iloc[:, len(data.columns) - 1]

    # Add zero-mean Gaussian noise of standard deviation 2 to all feature values
    np.random.seed(0)
    X = X + np.random.normal(0, 2, X.shape)
    X = (X + abs(X)) / 2

    # Normalize all feature vectors to have zero mean and unit variance
    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=12)
    X_new = pca.fit_transform(X)

    # Scree plot
    plt.ylabel("Eigenvalues")
    plt.xlabel("# of Features")
    plt.title("Scree plot")
    plt.ylim(0, max(pca.explained_variance_))

    plt.plot(pca.explained_variance_)
    plt.grid()
    plt.show()

    # Variance plot
    variance = np.cumsum(pca.explained_variance_ratio_ * 100)

    plt.ylabel("% Variance Explained")
    plt.xlabel("# of Features")
    plt.title("Variance plot")
    plt.ylim(min(variance), 100.5)
    plt.axhline(y=95, color="r", linestyle="--")

    plt.plot(variance)
    plt.grid()
    plt.show()

    # Scatter plots
    plot(X_new[:, 0], X_new[:, 1], y, "First", "Second")
    plot(X_new[:, 0], X_new[:, 2], y, "First", "Third")
    plot(X_new[:, 1], X_new[:, 2], y, "Second", "Third")

    # Loading matrix
    loadings = pca.components_.T

    idx = data.iloc[:, 0: len(data.columns) - 1].columns
    col = []
    for i in range(1, 13):
        col.append("PC%d" % i)

    loading_matrix = pd.DataFrame(loadings, columns=col, index=idx)
    loading_matrix.to_csv("loading_matrix.csv")

    plot_2(X[:, 0], X[:, 1], y)
