import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from Project import main

if __name__ == "__main__":
    vgg = main.complete_data_frame("color_vgg_16.csv")

    vgg = vgg[vgg.label != "martensite"]
    vgg = vgg[vgg.label != "pearlite+widmanstatten"]
    vgg = vgg[vgg.label != "spheroidite+widmanstatten"]

    x, y = main.split_data(vgg)
    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, stratify=y)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Default
    lower_error = 1
    best_max_pool = -1
    clf = DecisionTreeClassifier()

    for m in range(5):
        scores = cross_val_score(clf, x_train["max_pooling_%d" % (m + 1)].apply(pd.Series), y_train, cv=10)
        cross_val_error = 1 - np.average(scores)
        print("cv max_pooling_%d %f" % (m + 1, cross_val_error))

        if cross_val_error < lower_error:
            best_max_pool = m
            lower_error = cross_val_error

    clf.fit(x_train["max_pooling_%d" % (best_max_pool + 1)].apply(pd.Series), y_train)
    score = clf.score(x_test["max_pooling_%d" % (best_max_pool + 1)].apply(pd.Series), y_test)

    print("-----------------\n"
          "max_pooling_%d %f\n"
          "-----------------\n" % (best_max_pool + 1, 1 - score))
