import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    plt.ylabel("% error")
    plt.xlabel("Max pool layer")
    plt.title("LDA Cross validation error")

    plt.xlim([1, 5])
    plt.xticks(np.arange(1, 6, step=1))

    plt.plot([0,
              0.147039 * 100,
              0.107906 * 100,
              0.080090 * 100,
              0.243310 * 100,
              0.321920 * 100], marker='o')

    plt.plot([0,
              0.145399 * 100,
              0.098070 * 100,
              0.076811 * 100,
              0.232073 * 100,
              0.299048 * 100], marker='o')

    plt.legend(["VGG16", "VGG19"])
    plt.show()

    plt.ylabel("% error")
    plt.xlabel("Max pool layer")
    plt.title("DTC Cross validation error")

    plt.xlim([1, 5])
    plt.xticks(np.arange(1, 6, step=1))

    plt.plot([0,
              0.364199 * 100,
              0.248361 * 100,
              0.161740 * 100,
              0.117689 * 100,
              0.151957 * 100], marker='o')

    plt.plot([0,
              0.390481 * 100,
              0.263115 * 100,
              0.145505 * 100,
              0.112797 * 100,
              0.146986 * 100], marker='o')

    plt.legend(["VGG16", "VGG19"])
    plt.show()

    plt.ylabel("% error")
    plt.xlabel("Max pool layer")
    plt.title("XGB Cross validation error")

    plt.xlim([1, 5])
    plt.xticks(np.arange(1, 6, step=1))

    plt.plot([0,
              0.243522 * 100,
              0.133977 * 100,
              0.062136 * 100,
              0.042517 * 100,
              0.057218 * 100], marker='o')

    plt.plot([0,
              0.263009 * 100,
              0.120941 * 100,
              0.057166 * 100,
              0.049022 * 100,
              0.055526 * 100], marker='o')

    plt.legend(["VGG16", "VGG19"])
    plt.show()
