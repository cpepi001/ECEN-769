import pandas as pd


def return_a_complete_data_frame(file):
    image = pd.read_csv(file, header=None, usecols=[0], names=["image"])

    range_1 = [i for i in range(1, 65)]
    temp_max_pooling_1 = pd.read_csv(file, header=None, usecols=range_1)

    range_2 = [i for i in range(65, 193)]
    temp_max_pooling_2 = pd.read_csv(file, header=None, usecols=range_2)

    range_3 = [i for i in range(193, 449)]
    temp_max_pooling_3 = pd.read_csv(file, header=None, usecols=range_3)

    range_4 = [i for i in range(449, 961)]
    temp_max_pooling_4 = pd.read_csv(file, header=None, usecols=range_4)

    range_5 = [i for i in range(961, 1473)]
    temp_max_pooling_5 = pd.read_csv(file, header=None, usecols=range_5)

    label = pd.read_csv(file, header=None, usecols=[1473], names=["label"])

    max_pooling_1 = pd.DataFrame()
    max_pooling_2 = pd.DataFrame()
    max_pooling_3 = pd.DataFrame()
    max_pooling_4 = pd.DataFrame()
    max_pooling_5 = pd.DataFrame()

    max_pooling_1["max_pooling_1"] = temp_max_pooling_1.values.tolist()
    max_pooling_2["max_pooling_2"] = temp_max_pooling_2.values.tolist()
    max_pooling_3["max_pooling_3"] = temp_max_pooling_3.values.tolist()
    max_pooling_4["max_pooling_4"] = temp_max_pooling_4.values.tolist()
    max_pooling_5["max_pooling_5"] = temp_max_pooling_5.values.tolist()

    return pd.concat(
        [image, max_pooling_1, max_pooling_2, max_pooling_3, max_pooling_4, max_pooling_5, label], axis=1)
