import os

import numpy as np
import pandas as pd
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image


"""
    [x] spheroidite(374)                -> 100 | 274
    [x] network(212)                    -> 100 | 112
    [x] pearlite(124)                   -> 100 | 24
    [x] pearlite+spheroidite(107)       ->   0 | 107
    [x] spheroidite+widmanstatten(81)   ->  60 | 21
    [x] martensite(36)                  ->   0 | 36
    [x] pearlite+widmanstatten(27)      ->   0 | 27
                                        -> 360 | 601 = 961
"""

if __name__ == "__main__":
    data = pd.read_csv("micrograph.csv")

    # Get the training labels
    primary_label = "primary_microconstituent"
    training_labels = ["spheroidite", "network", "pearlite", "spheroidite+widmanstatten"]
    data_to_split = data.loc[data[primary_label].isin(training_labels)]

    # Split the training data
    testing_data = pd.DataFrame()
    training_data = pd.DataFrame()
    for i in range(len(training_labels)):
        if i == 3:  # spheroidite+widmanstatten
            max_size = 60
        else:  # spheroidite, network, pearlite
            max_size = 100

        size = data_to_split.loc[data_to_split[primary_label] == training_labels[i]].shape[0]
        temp_training_data = data_to_split.loc[data_to_split[primary_label] == training_labels[i]].head(max_size)
        temp_testing_data = data_to_split.loc[data_to_split[primary_label] == training_labels[i]].tail(size - max_size)

        training_data = training_data.append(temp_training_data)
        testing_data = testing_data.append(temp_testing_data)

    # Add the rest labels
    testing_data = testing_data.append(data.loc[data[primary_label] == "pearlite+spheroidite"])
    testing_data = testing_data.append(data.loc[data[primary_label] == "martensite"])
    testing_data = testing_data.append(data.loc[data[primary_label] == "pearlite+widmanstatten"])

    base_model = VGG16(weights="imagenet", include_top=False)

    block_pools = []
    block_pools_shapes = []
    for b in range(0, 5):
        block_pools.append(Model(inputs=base_model.input, outputs=base_model.get_layer("block%d_pool" % (b + 1)).input))
        block_pools_shapes.append(block_pools[b].output_shape[3])

    iteration = 0
    features = {}
    for i, row in testing_data.iterrows():
        print("%d / %d" % (iteration, testing_data.shape[0]))
        filepath = os.path.join("images", row["path"])

        img = image.load_img(filepath)
        x = image.img_to_array(img)
        x = x[0:484, :, :]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        max_pooling_ranges = np.zeros([5, 2], dtype=int)

        idx = 0
        for r in range(len(block_pools_shapes)):
            max_pooling_ranges[r] = [idx, idx + block_pools_shapes[r]]
            idx += block_pools_shapes[r]

        max_pooling = np.zeros(idx)
        for b in range(len(block_pools)):
            xb = block_pools[b].predict(x)
            max_pooling[max_pooling_ranges[b][0]: max_pooling_ranges[b][1]] = np.mean(xb, axis=(0, 1, 2))

        features[iteration] = [row["path"], max_pooling, row[primary_label]]
        iteration += 1

    with open("testing_data.csv", "w") as f:
        for key in features.keys():
            f.write("%s," % features[key][0])
            for v in range(len(features[key][1])):
                f.write("%f," % features[key][1][v])
            f.write("%s\n" % features[key][2])
