import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

import utils


def split_data(data):
    x = data.iloc[:, 1: len(data.columns) - 1]
    y = data.iloc[:, len(data.columns) - 1]

    return x, y


def multiple_svm(train_x, train_y, test_x, test_y):
    svm = SVC()

    lower_error = 1
    best_max_pool = -1
    for m in range(5):
        scores = cross_val_score(svm, train_x["max_pooling_%d" % (m + 1)].apply(pd.Series), train_y, cv=10)
        cross_val_error = 1 - np.average(scores)
        print("max_pooling_%d %f" % (m + 1, cross_val_error))

        if cross_val_error < lower_error:
            best_max_pool = m
            lower_error = cross_val_error
    print()

    best_max_pool += 1
    print("predicting with the best max_pooling layer which is %d" % best_max_pool)

    best_svm = SVC()
    best_svm.fit(train_x["max_pooling_%d" % best_max_pool].apply(pd.Series), train_y)
    best_score = best_svm.score(test_x["max_pooling_%d" % best_max_pool].apply(pd.Series), test_y)

    print("test error ", (1 - best_score))
    print()


if __name__ == "__main__":
    training_labels = ["spheroidite", "network", "pearlite", "spheroidite+widmanstatten"]

    testing_data = utils.return_a_complete_data_frame("testing_data.csv")
    training_data = utils.return_a_complete_data_frame("training_data.csv")

    testing_data_4 = testing_data.loc[testing_data["label"].isin(training_labels)]

    # Part a, b #
    for i in range(len(training_labels)):
        for j in range(i + 1, len(training_labels)):
            string = "%s v %s (averaged cross-validated error)" % (training_labels[i], training_labels[j])
            print(string)
            print("-" * len(string))

            temp_training = training_data.loc[training_data["label"].isin([training_labels[i], training_labels[j]])]
            temp_testing = testing_data_4.loc[testing_data_4["label"].isin([training_labels[i], training_labels[j]])]

            x_test, y_test = split_data(temp_testing)
            x_train, y_train = split_data(temp_training)

            multiple_svm(x_train, y_train, x_test, y_test)

    x_test, y_test = split_data(testing_data_4)
    x_train, y_train = split_data(training_data)

    ovo_classifier = OneVsOneClassifier(SVC())
    for i in range(1, 6):
        ovo_classifier.fit(x_train["max_pooling_%d" % i].apply(pd.Series), y_train)
        score = ovo_classifier.score(x_test["max_pooling_%d" % i].apply(pd.Series), y_test)
        print("Predicting with OneVsOne Classifier using max_pooling_%d" % i)
        print(1 - score)

    # Part c #
    print("part c")
    pearlite_spheroidite_test = testing_data.loc[testing_data["label"].isin(["pearlite+spheroidite"])]
    x_test, y_test = split_data(pearlite_spheroidite_test)

    pearlite_spheroidite_svm = SVC()

    pearlite_spheroidite_train = training_data.loc[training_data["label"].isin(["pearlite", "spheroidite"])]
    x_train, y_train = split_data(pearlite_spheroidite_train)

    pearlite_spheroidite_svm.fit(x_train["max_pooling_5"].apply(pd.Series), y_train)

    ovo_classifier = OneVsOneClassifier(SVC())

    x_train, y_train = split_data(training_data)

    ovo_classifier.fit(x_train["max_pooling_5"].apply(pd.Series), y_train)

    y_pred_svm = pearlite_spheroidite_svm.predict(x_test["max_pooling_5"].apply(pd.Series))
    y_pred_ovo = ovo_classifier.predict(x_test["max_pooling_5"].apply(pd.Series))

    for i in range(len(y_pred_svm)):
        print("%s | %s, %s" % ((i + 1), y_pred_svm[i], y_pred_ovo[i]))

    # Part d #
    print("part d")
    martensite_test = testing_data.loc[testing_data["label"].isin(["martensite"])]

    x_test, _ = split_data(martensite_test)
    x_train, y_train = split_data(training_data)

    ovo_classifier = OneVsOneClassifier(SVC())
    ovo_classifier.fit(x_train["max_pooling_5"].apply(pd.Series), y_train)

    martensite_pred_ovo = ovo_classifier.predict(x_test["max_pooling_5"].apply(pd.Series))

    pearlite_widmanstatten_test = testing_data.loc[testing_data["label"].isin(["pearlite+widmanstatten"])]

    x_test, _ = split_data(pearlite_widmanstatten_test)
    x_train, y_train = split_data(training_data)

    ovo_classifier = OneVsOneClassifier(SVC())
    ovo_classifier.fit(x_train["max_pooling_5"].apply(pd.Series), y_train)

    pearlite_widmanstatten_pred_ovo = ovo_classifier.predict(x_test["max_pooling_5"].apply(pd.Series))

    for i in range(len(martensite_pred_ovo)):
        if i > 26:
            print("%s | %s, _" % ((i + 1), martensite_pred_ovo[i]))
        else:
            print("%s | %s, %s" % ((i + 1), martensite_pred_ovo[i], pearlite_widmanstatten_pred_ovo[i]))

