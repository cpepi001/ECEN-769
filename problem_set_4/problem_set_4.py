from itertools import combinations

import numpy as np
import pandas as pd
from clang.cindex import xrange
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    SFE_orig = pd.read_table("Stacking_Fault_Energy_Dataset.txt")

    # RSS = ((y_true - y_pred)**2).sum()
    # TSS = ((y_true - y_true.mean())**2).sum()
    # R_square = (TSS - RSS) / TSS
    # R_square_adj = 1 - (((1 - R_square) * (n - 1)) / (n - d - 1))

    # pre-process the data
    n_orig = SFE_orig.shape[0]  # original number of rows
    p_orig = np.sum(SFE_orig > 0) / n_orig  # fraction of nonzero entries for each column
    SFE_colnames = SFE_orig.columns[p_orig > 0.6]
    SFE_col = SFE_orig[SFE_colnames]  # throw out columns with fewer than 60% nonzero entries
    m_col = np.prod(SFE_col, axis=1)
    SFE = SFE_col[m_col != 0]  # throw out rows that contain any zero entries

    x = SFE.iloc[:, :-1]
    yr = SFE["SFE"]

    model = LinearRegression()

    for feat in x.columns:
        xr = np.array(SFE[feat])
        xrr = xr.reshape((-1, 1))  # format xr for Numpy regression code
        fig = plt.figure(figsize=(8, 6), dpi=150)
        plt.style.use("seaborn")
        plt.xlabel(feat, size=24)
        plt.ylabel("SFE", size=24)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.scatter(xr, yr, s=32, marker="o", facecolor="none", edgecolor="r", linewidth=1.5)
        model.fit(xrr, yr)
        y_pred = model.predict(xrr)
        plt.plot(xrr, model.predict(xrr), c="green", lw=2)
        plt.show()

        RSS = ((yr - y_pred) ** 2).sum() / len(yr)

        print(feat, model.coef_[0], model.intercept_, RSS, model.score(xrr, yr))

    # Exhaustive search
    best_predictors = {}
    variables = x.columns
    for i in xrange(1, len(variables) + 1):
        flag = 0
        combs = [list(x) for x in combinations(variables, i)]
        for lst in combs:
            xr = np.array(SFE[lst])
            xrr = xr.reshape((-1, len(lst)))
            model.fit(xrr, yr)
            y_pred = model.predict(xrr)

            RSS = ((yr - y_pred) ** 2).sum()
            TSS = ((yr - yr.mean()) ** 2).sum()

            R_square = (TSS - RSS) / TSS
            R_square_adj = 1 - (((1 - R_square) * (len(yr) - 1)) / (len(yr) - len(lst) - 1))

            if flag == 0:
                flag = 1
                best_predictors[i] = ["_".join(lst), RSS / len(yr), R_square, R_square_adj]
            else:
                if R_square_adj > best_predictors.get(i)[-1]:
                    best_predictors[i] = ["_".join(lst), RSS / len(yr), R_square, R_square_adj]

    print("Exhaustive search")
    for predictor in best_predictors:
        print(best_predictors.get(predictor))

    # Sequential forward search
    best_elements = []
    for i in xrange(1, len(variables) + 1):
        best_element = -1
        variables = x.columns
        best_adj_r_square = -1
        combs = [list(x) for x in combinations(variables, 1)]
        for lst in combs:
            for e in lst:
                # print("processing element %s" % e)
                temp = best_elements.copy()
                temp.append(e)
                # print(temp)

                xr = np.array(SFE[temp])
                xrr = xr.reshape((-1, len(temp)))
                model.fit(xrr, yr)
                y_pred = model.predict(xrr)

                RSS = ((yr - y_pred) ** 2).sum()
                TSS = ((yr - yr.mean()) ** 2).sum()

                R_square = (TSS - RSS) / TSS
                R_square_adj = 1 - (((1 - R_square) * (len(yr) - 1)) / (len(yr) - len(temp) - 1))

                # print("adj r2 for element %s is %f" % (e, R_square_adj))

                if R_square_adj > best_adj_r_square:
                    best_element = e
                    best_adj_r_square = R_square_adj

        best_elements.append(best_element)
        x = x.drop([best_element], axis=1)
        # print("dropping %s" % best_element)

    best_predictors = {}
    for i in range(1, len(best_elements) + 1):
        lst = best_elements[0:i]
        xr = np.array(SFE[lst])
        xrr = xr.reshape((-1, len(lst)))
        model.fit(xrr, yr)
        y_pred = model.predict(xrr)

        RSS = ((yr - y_pred) ** 2).sum()
        TSS = ((yr - yr.mean()) ** 2).sum()

        R_square = (TSS - RSS) / TSS
        R_square_adj = 1 - (((1 - R_square) * (len(yr) - 1)) / (len(yr) - len(lst) - 1))

        best_predictors[i] = ["_".join(lst), RSS / len(yr), R_square, R_square_adj]

    print("Sequential forward search")
    for predictor in best_predictors:
        print(best_predictors.get(predictor))
