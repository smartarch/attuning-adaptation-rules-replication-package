#!/usr/bin/env python3
import argparse
import math
import h5py
import tensorflow as tf
import re
from tensorflow.keras import layers, activations


def loadData(fileName, batchSize):
    with h5py.File(fileName, 'r') as hf:
        trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'])).cache()
        valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'])).cache()

    trainDS = trainDS.shuffle(buffer_size=len(trainDS)).batch(batchSize).prefetch(1000)
    valDS = valDS.batch(batchSize).prefetch(1000)

    return (trainDS, valDS)


class BelowThreshold(layers.Layer):
    def __init__(self, name, minValue, maxValue):
        super().__init__(name=name)
        self.offset = tf.constant(-minValue, dtype=tf.float32)
        self.scale = tf.constant(1 / (maxValue - minValue), dtype=tf.float32)
        self.threshold = tf.Variable(tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32), trainable=True)
        print(f'{name} ... {self.offset} {self.scale}')

    def call(self, y, training=False):
        y = (self.threshold - (y + self.offset) * self.scale) * 10
        y = activations.sigmoid(y)
        return y


class AboveThreshold(layers.Layer):
    def __init__(self, name, minValue, maxValue):
        super().__init__(name=name)
        self.offset = tf.constant(-minValue, dtype=tf.float32)
        self.scale = tf.constant(1 / (maxValue - minValue), dtype=tf.float32)
        self.threshold = tf.Variable(tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32), trainable=True)
        print(f'{name} ... {self.offset} {self.scale}')

    def call(self, y, training=False):
        y = ((y + self.offset) * self.scale - self.threshold) * 10
        y = activations.sigmoid(y)
        return y


class CorrectTime(layers.Layer):
    def __init__(self, name, minValue, maxValue, capacity, sigma=None):
        super().__init__(name=name)

        if sigma is None:
            sigma = (maxValue - minValue) / capacity

        # random static points
        self.refPoints = tf.random.uniform(shape=(capacity,), minval=minValue, maxval=maxValue, dtype=tf.float32)

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y, training=False):
        y = tf.exp(-tf.square(self.refPoints - y) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectPlace(layers.Layer):
    def __init__(self, name, minPos, maxPos, capacity, sigma=None):
        super().__init__(name=name)

        numPoints = capacity ** 2

        if sigma is None:
            sigma = math.sqrt((maxPos[0] - minPos[0]) * (maxPos[1] - minPos[1]) / numPoints)

        self.refPoints = tf.stack(
            [
                tf.random.uniform(shape=(numPoints,), minval=minPos[0], maxval=maxPos[0], dtype=tf.float32),
                tf.random.uniform(shape=(numPoints,), minval=minPos[1], maxval=maxPos[1], dtype=tf.float32)
            ],
            axis=1
        )

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y):
        y = tf.expand_dims(y, axis=1)
        y = tf.exp(-tf.reduce_sum(tf.square(self.refPoints - y), axis=2) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectEvent(layers.Layer):
    def __init__(self, name, capacity):
        super().__init__(name=name)
        self.dense = layers.Dense(name=f'{name}_dense_last', units=1)
        self.hidden = None
        if capacity > 1:
            self.hidden = layers.Dense(name=f'{name}_dense_hidden', units=capacity, activation='relu')

    def call(self, y):
        if self.hidden is not None:
            y = self.hidden(y)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class IndustryModel(tf.keras.Model):
    def __init__(self, args):
        super(IndustryModel, self).__init__()
        self.fromTime = 2400
        self.toTime = 33600
        self.gatePositions = [
            [178.64561, 98.856514],
            [237.03545, 68.872505],
            [237.0766, 135.65627],
        ]

        self.trainTime = args.time_capacity > 0
        self.trainPlace = args.place_capacity > 0
        self.trainEvents = args.event_capacity > 0
        self.timeAboveBelow = args.time_above_below
        if self.trainTime:
            if args.time_above_below:
                self.timeLow = AboveThreshold('timeLow', minValue=0, maxValue=36000)
                self.timeHigh = BelowThreshold('timeHigh', minValue=0, maxValue=36000)
            else:
                self.time = CorrectTime('time', minValue=0, maxValue=36000, capacity=args.time_capacity)
        if self.trainPlace:
            self.placeA = CorrectPlace('placeA', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
            self.placeB = CorrectPlace('placeB', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
            self.placeC = CorrectPlace('placeC', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
        if self.trainEvents:
            self.headGear = CorrectEvent('headGear', capacity=args.event_capacity)

    def atGate(self, id, x, y):
        gatePos = self.gatePositions[id]
        dx = x - gatePos[0]
        dy = y - gatePos[1]
        return tf.math.sqrt(dx * dx + dy * dy) <= 10

    def call(self, y):
        if self.trainTime:
            if self.timeAboveBelow:
                yTime = self.timeLow(y[:, 0:1]) + self.timeHigh(y[:, 0:1])
            else:
                yTime = self.time(y[:, 0:1]) * 2  # 2x compentsates for other branches
        else:
            yTime = tf.cast(y[:, 0:1] >= self.fromTime, dtype=tf.float32) + tf.cast(y[:, 0:1] <= self.toTime, dtype=tf.float32)

        if self.trainPlace:
            yPlaceA = self.placeA(y[:, 1:3]) * y[:, 3:4]
            yPlaceB = self.placeB(y[:, 1:3]) * y[:, 4:5]
            yPlaceC = self.placeC(y[:, 1:3]) * y[:, 5:6]
        else:
            yPlaceA = tf.cast(tf.logical_and(self.atGate(0, y[:, 1:2], y[:, 2:3]), y[:, 3:4] == 1), dtype=tf.float32)
            yPlaceB = tf.cast(tf.logical_and(self.atGate(1, y[:, 1:2], y[:, 2:3]), y[:, 4:5] == 1), dtype=tf.float32)
            yPlaceC = tf.cast(tf.logical_and(self.atGate(2, y[:, 1:2], y[:, 2:3]), y[:, 5:6] == 1), dtype=tf.float32)

#        yPlace = activations.hard_sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 100)  # OR
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)  # OR

        if self.trainEvents:
            yHeadGear = self.headGear(y[:, 6:8])
        else:
            yHeadGear = tf.cast(tf.logical_and(y[:, 6:7] == 1, y[:, 7:8] == 0), dtype=tf.float32)

#        return activations.hard_sigmoid((yTime + yPlace + yHeadGear - 3.5) * 100)  # AND (time already represents 2)
        return activations.sigmoid((yTime + yPlace + yHeadGear - 3.5) * 10)  # AND (time already represents 2)


class SynthModel(tf.keras.Model):
    def __init__(self, args, points):
        super(SynthModel, self).__init__()
        self.pointLayer = [
            CorrectTime(name=f'points_{idx}', minValue=0, maxValue=1, capacity=args.time_capacity) for idx in range(points)
        ]

    def call(self, y):
        sum = 0.0
        for idx in range(len(self.pointLayer)):
            sum = sum + self.pointLayer[idx](y[:, idx:(idx + 1)])

        correction = float(len(self.pointLayer)) - 0.5
        return activations.sigmoid((sum - correction) * 10000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument("--learning_rate_decay", default=True, action="store_true", help="Decay learning rate")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--time_capacity", default=20, type=int, help="Learning capacity of time layers.")
    parser.add_argument("--place_capacity", default=20, type=int, help="Learning capacity of place layers.")
    parser.add_argument("--event_capacity", default=1, type=int, help="Learning capacity of event layers.")
    parser.add_argument("--time_above_below", default=False, action="store_true", help="Switch to intreval (above/below) training of time predicate.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Fitting and evaluation will be verbose.")
    parser.add_argument("--model", type=str, default='industry', help="Selected model ('industry' or 'synth').")
    parser.add_argument("--model_name", type=str, default='I', help="Name of the model (just for the output).")
    args = parser.parse_args()

    # Use given number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    trainDS, valDS = loadData(args.input_file, args.batch_size)

    if args.model == 'industry':
        model = IndustryModel(args)
    elif args.model == 'synth':
        model = SynthModel(args, trainDS.element_spec[0].shape[1])
    else:
        print("Invalid model selecte {}".formt(args.model))
        exit(1)

    learning_rate = args.learning_rate
    if args.learning_rate_decay:
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, args.epochs * len(trainDS))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    if args.verbose:
        print("File: {}, training size: {}, validation size: {}".format(
            args.input_file, len(trainDS), len(valDS)))
        print("Batch size: {}, epochs: {}, learning rate: {}, learning decay: {}, label smoothing: {}".format(
            args.batch_size, args.epochs, args.learning_rate, args.learning_rate_decay, args.label_smoothing))
        print("Time learning capacity: {}, place learning capacity: {}, events learning capacity: {}".format(
            args.time_capacity, args.place_capacity, args.event_capacity))
        model.build(input_shape=trainDS.element_spec[0].shape)
        model.summary()

    accuracies = []
    for epochIdx in range(0, args.epochs):
        model.fit(trainDS, epochs=1, verbose=args.verbose)
        evalRes = model.evaluate(valDS, return_dict=True, verbose=args.verbose)
        accuracies.append(evalRes['accuracy'])

    bareFileName = re.sub(r"^.*/", "", args.input_file)
    print("{};smart-{};{}".format(bareFileName, args.model_name, ";".join(map(str, accuracies))))
