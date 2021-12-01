import os

import numpy as np
import pandas as pd
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image


def split_data(data):
    x = data.iloc[:, 1: len(data.columns) - 1]
    y = data.iloc[:, len(data.columns) - 1]

    return x, y


def complete_data_frame(file):
    id = pd.read_csv(file, header=None, usecols=[0], names=["id"])

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
        [id, max_pooling_1, max_pooling_2, max_pooling_3, max_pooling_4, max_pooling_5, label], axis=1)


def write_data(file_name, data):
    with open(file_name, "w") as f:
        for key in data.keys():
            f.write("%s," % data[key][0])
            for v in range(len(data[key][1])):
                f.write("%f," % data[key][1][v])
            f.write("%s\n" % data[key][2])


def get_block_pools(vgg_model):
    if vgg_model == "VGG16":
        model = VGG16(weights="imagenet", include_top=False)
    else:
        model = VGG19(weights="imagenet", include_top=False)

    block_pools = []
    block_pools_shapes = []

    for b in range(0, 5):
        block_pools.append(Model(inputs=model.input, outputs=model.get_layer("block%d_pool" % (b + 1)).input))
        block_pools_shapes.append(block_pools[b].output_shape[3])

    return block_pools, block_pools_shapes


def get_feature(target, block_pool, block_pool_shape):
    idx = 0
    for r in range(len(block_pool_shape)):
        max_pooling_ranges[r] = [idx, idx + block_pool_shape[r]]
        idx += block_pool_shape[r]

    max_pooling = np.zeros(idx)
    for b in range(len(block_pool)):
        xb = block_pool[b].predict(target)
        max_pooling[max_pooling_ranges[b][0]: max_pooling_ranges[b][1]] = np.mean(xb, axis=(0, 1, 2))

    return [row.micrograph_id, max_pooling, row[primary_label]]


if __name__ == "__main__":
    df = pd.read_csv("micrograph.csv")

    # Primary label
    primary_label = "primary_microconstituent"

    # Find all the classes
    classes = df[primary_label].unique()

    # Get the block pool layers
    bp_16, bps_16 = get_block_pools("VGG16")
    bp_19, bps_19 = get_block_pools("VGG19")

    # Do the magic
    features_16 = {}
    features_19 = {}
    for i, row in df.iterrows():
        # Construct the path of the image
        image_path = os.path.join("images", row.path)

        # Process the image
        img = image.load_img(image_path)
        x = image.img_to_array(img)
        x = x[0:484, :, :]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        max_pooling_ranges = np.zeros([5, 2], dtype=int)

        features_16[i] = get_feature(x, bp_16, bps_16)
        features_19[i] = get_feature(x, bp_19, bps_19)

        print("%d / %d" % (i, df.shape[0]))

    write_data("vgg_16.csv", features_16)
    write_data("vgg_19.csv", features_19)
