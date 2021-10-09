""""
Foundations of Pattern Recognition and Machine Learning
Chapter 1 Figure 1.5
Author: Ulisses Braga-Neto

Plots histogram, density plots, and 2-D LDA classifier for Stacking Fault Energy data set
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


def estimate_error(pX, pY):
    y_pred = clf.predict(pX)

    error = 0
    counter = 0
    for y in pY:
        if y != y_pred[counter]:
            error += 1
        counter += 1

    return error / len(pY)


def plotty(pX, pY, pA, pB, pName, pred, title):
    var1 = pred[0]
    var2 = pred[1]

    plt.style.use('seaborn')
    plt.figure(figsize=(8, 8), dpi=150)
    plt.title(title)
    plt.axis('equal')
    plt.scatter(pX[~pY, 0], pX[~pY, 1], c='blue', s=10, label='Low SFE')
    plt.scatter(pX[pY, 0], pX[pY, 1], c='orange', s=10, label='High SFE')
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    plt.plot([left, right], [-left * pA[0] / pA[1] - pB / pA[1], -right * pA[0] / pA[1] - pB / pA[1]], 'k', linewidth=2)
    plt.xlim(left, right)
    plt.ylim(bottom, top)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(var1, fontsize=18)
    plt.ylabel(var2, fontsize=18)
    plt.legend(fontsize=14, loc="lower left", markerfirst=False, markerscale=1.5, handletextpad=0.1)
    plt.savefig('c01_matex-%s.png' % pName, bbox_inches='tight')


if __name__ == "__main__":
    SFE_data = pd.read_table('Stacking_Fault_Energy_Dataset.txt')

    # pre-process the data
    f_org = SFE_data.columns[:-1]  # original features
    n_org = SFE_data.shape[0]  # original number of training points
    p_org = np.sum(SFE_data.iloc[:, :-1] > 0) / n_org  # fraction of nonzero components for each feature
    f_drp = f_org[p_org < 0.6]  # features with less than 60% nonzero components
    SFE1 = SFE_data.drop(f_drp, axis=1)  # drop those features
    s_min = SFE1.min(axis=1)
    SFE2 = SFE1[s_min != 0]  # drop sample points with any zero values
    SFE = SFE2[(SFE2.SFE < 35) | (SFE2.SFE > 45)]  # drop sample points with middle responses

    ####################################################################################################################

    SFE_headers = list(SFE.columns.values)
    SFE_headers = SFE_headers[0: len(SFE_headers) - 1]

    train_data, test_data = train_test_split(SFE, train_size=0.2, shuffle=False)

    train_data_low_SFE = train_data[train_data["SFE"] <= 35]
    train_data_high_SFE = train_data[train_data["SFE"] >= 45]

    predictors = {}
    for i in range(len(SFE_headers)):
        stat, p = ttest_ind(train_data_low_SFE[SFE_headers[i]], train_data_high_SFE[SFE_headers[i]], equal_var=False)
        predictors["%s" % (SFE_headers[i])] = [abs(stat), p]

    predictors = sorted(predictors.items(), key=lambda v: v[1][0], reverse=True)

    print("Element | T statistic | p-value")
    print("--------+-------------+--------")
    for predictor in predictors:
        print(f"{predictor[0]:<2}      | {predictor[1][0]:.4f}      | {predictor[1][1]:.4f}")
    print()

    ####################################################################################################################

    Y_2 = test_data.SFE >= 45
    Y_1 = train_data.SFE >= 45

    y_true = 0
    y_false = 0
    for y in Y_1:
        if y:
            y_true += 1
        else:
            y_false += 1

    priors = [y_true / len(Y_1), y_false / len(Y_1)]
    clf = LDA(priors=priors)

    test_error = []
    train_error = []
    for i in range(2, 8, 1):
        top_predictors = []
        for j in range(i):
            top_predictors.append(predictors[j][0])

        X_1 = train_data[top_predictors].values

        clf.fit(X_1, Y_1.astype(int))

        a = clf.coef_[0]
        b = clf.intercept_[0]

        X_2 = test_data[top_predictors].values

        if len(top_predictors) == 2:
            plotty(X_1, Y_1, a, b, "c", top_predictors, "Training data")
            plotty(X_2, Y_2, a, b, "d", top_predictors, "Test data")

        test_error.append(estimate_error(X_2, Y_2))
        train_error.append(estimate_error(X_1, Y_1))

    print("Train error | Test error | # Predictors")
    print("------------+------------+-------------")
    for i in range(len(test_error)):
        print(f"{train_error[i]:.4f}      | {test_error[i]:.4f}     | {i + 2:}")
