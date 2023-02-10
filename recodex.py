import sys
import h5py
import re
import tensorflow as tf
from tensorflow.keras import layers, activations

verbose = 1

categories_sizes = []


def loadData(fileName, batchSize):
    with h5py.File(fileName, 'r') as hf:
        for i in range(0, 5):
            categories_sizes.append(int(max([max(hf['train-inputs'][:, i]), max(hf['val-inputs'][:, i])])))

        trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'])).cache()
        valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'])).cache()

    trainDS = trainDS.shuffle(buffer_size=500).batch(batchSize).prefetch(500)
    valDS = valDS.batch(batchSize).prefetch(500)

    return (trainDS, valDS)


def get_normalization_layer(idx, dataset):
    normalizer = tf.keras.layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, _: x[idx])
    normalizer.adapt(feature_ds)
    return normalizer


def get_category_encoding_layer(idx):
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=categories_sizes[idx] + 1, output_mode="one_hot")
    return lambda feature: encoder(feature)


def prepare_inputs(trainDS):
    all_inputs = tf.keras.Input(shape=(12), dtype='float32')
    encoded_features = []

    for idx in range(1, 5):
        col = tf.cast(all_inputs[:, idx], dtype='int32')
        encoding_layer = get_category_encoding_layer(idx)
        encoded_col = encoding_layer(col)
        encoded_features.append(encoded_col)

    for idx in range(5, 8):
        col = all_inputs[:, idx:(idx + 1)]
        normalization_layer = get_normalization_layer(idx, trainDS)
        encoded_col = normalization_layer(col)
        encoded_features.append(encoded_col)

    for idx in range(8, 12):
        encoded_features.append(all_inputs[:, idx:(idx + 1)])

    encoded_features.append(all_inputs[:, 5:6])
    encoded_features.append(all_inputs[:, 7:8])

    return all_inputs, encoded_features


class SkipExtraFeaturesLayer(layers.Layer):
    def __init__(self):
        super().__init__(name='skip_last_feature')

    def call(self, y):
        return y[:, :-2]


class CustomLayer(layers.Layer):
    def __init__(self, capacity):
        super().__init__(name='custom')
        self.hidden = layers.Dense(name='base_dense_hidden', units=capacity, activation='relu')
        self.last = layers.Dense(name='last_dense', units=4)

    def call(self, y):
        ref = y[:, -1:]  # last feature is not normalized time limits
        isShort = tf.cast(ref <= 60.0, dtype=tf.float32)  # 1 = is short, 0 = is long

        y = self.hidden(y[:, :-2])
        y = self.last(y)
        y = tf.concat([
            y[:, 0:1] + 1.0 - isShort,  # if long, +1 is added to first category
            y[:, 1:2] * isShort,  # if long, remaining categories are zeroed
            y[:, 2:3] * isShort,
            y[:, 3:4] * isShort,
        ], axis=1)
        return activations.softmax(y)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("At least 2 args expected (file name and one layer witdth)")
        sys.exit(0)

    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    fileName = sys.argv[1]
    trainDS, valDS = loadData(fileName, 100)

    all_inputs, encoded_features = prepare_inputs(trainDS)

    last_layer = tf.keras.layers.Concatenate()(encoded_features)

    if sys.argv[2] == 'custom':
        if verbose == 1:
            print("Using custom layer...")
        last_layer = CustomLayer(int(sys.argv[3]))(last_layer)
    else:
        last_layer = SkipExtraFeaturesLayer()(last_layer)
        for width in sys.argv[2:]:
            last_layer = tf.keras.layers.Dense(int(width), activation='relu')(last_layer)
        last_layer = tf.keras.layers.Dense(4, activation='softmax')(last_layer)

    model = tf.keras.Model(all_inputs, last_layer)

    learning_rate = tf.keras.experimental.CosineDecay(0.01, 100 * len(trainDS))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    if verbose == 1:
        model.build(input_shape=trainDS.element_spec[0].shape)
        model.summary()

    accuracies = []
    for epochIdx in range(0, 100):
        model.fit(trainDS, epochs=1, verbose=verbose)
        evalRes = model.evaluate(valDS, return_dict=True, verbose=verbose)
        accuracies.append(evalRes['accuracy'])

    bareFileName = re.sub(r"^.*/", "", fileName)
    print("{};dense-{};{}".format(bareFileName, "-".join(sys.argv[2:]), ";".join(map(str, accuracies))))
