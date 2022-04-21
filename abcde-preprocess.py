#!/usr/bin/env python3

import argparse
import h5py
import random
import csv
import statistics
from progress.bar import Bar
import numpy as np


def load_csv_data(args):
    limit = args.limit
    if limit == 0:
        return []

    result = []
    with open(args.csv_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=args.delimiter)
        for line in reader:
            if limit <= 0:
                break

            limits = float(line["limits"])
            duration = float(line["duration"])
            if duration > limits * args.sus_threshold:
                continue

            result.append(line)
            limit = limit - 1

    # append computed history colums (has to be done before suffling)
    if args.history:
        history = {}  # [user][exercise][runtime] = [ ... durations ... ]
        for line in result:
            user = line["user_id"]
            exercise = line["exercise_id"]
            runtime = line["runtime_id"]
            if user not in history:
                history[user] = {}
            if exercise not in history[user]:
                history[user][exercise] = {}
            if runtime not in history[user][exercise]:
                history[user][exercise][runtime] = []

            attempt = len(history[user][exercise][runtime])
            line["attempt"] = attempt
            if attempt > 0:
                line["previous"] = history[user][exercise][runtime][-1]
            else:
                line["previous"] = int(float(line["limits"]) * 1000.0)  # fallback if there is no history

            history[user][exercise][runtime].append(int(float(line["duration"]) * 1000.0))

    return result


def load_reference_solutions(args, data):
    refs = {}  # [exercise][runtime] = avg duration
    with open(args.refs, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=args.delimiter)
        for line in reader:
            exercise = line["exercise_id"]
            runtime = line["runtime_id"]
            if exercise not in refs:
                refs[exercise] = {}
            if runtime not in refs[exercise]:
                refs[exercise][runtime] = []
            refs[exercise][runtime].append(float(line["duration"]))

    for exercise in refs:
        for runtime in refs[exercise]:
            refs[exercise][runtime] = statistics.mean(refs[exercise][runtime])

    for line in data:
        exercise = line["exercise_id"]
        runtime = line["runtime_id"]
        if exercise not in refs:
            line["ref_duration"] = float(line["limits"])
        elif runtime not in refs[exercise]:
            line["ref_duration"] = list(refs[exercise].values())[0]  # pick the first runtime instead
        else:
            line["ref_duration"] = refs[exercise][runtime]


# Translators convert the string values from CSV to final int values applying various transformations


# Simple str to int converter
class IntTranslator:
    def translate(self, value):
        return int(value)


# Simple str to float converter
class FloatTranslator:
    def translate(self, value):
        return float(value)


# Translates duration value (float) into a bucket index, whilst buckets are separated by exact marks
# E.g., marks [1,5] will create 3 buckets (lower than 1s, 1-5s, over 5s)
class DurationsTranslator:
    marks = []

    # initialize the translator with specific marks (that determine buckets)
    def __init__(self, marks=[1, 5]):
        self.marks = marks

    def translate(self, value):
        i = 0
        f = float(value)
        while i < len(self.marks) and f > self.marks[i]:
            i += 1
        return i


# Helper class for translating string IDs into sequential numeric IDs
class HashTranslator:
    table = {}
    counter = 0

    def translate(self, value):
        if value not in self.table:
            self.counter = self.counter + 1
            self.table[value] = self.counter

        return self.table[value]


# Return a dictionary with translator instance for every column
def prepare_translators(duration_markers=None):
    res = {
        "solution_id": HashTranslator(),
        "group_id": HashTranslator(),
        "tlgroup_id": HashTranslator(),
        "exercise_id": HashTranslator(),
        "runtime_id": HashTranslator(),
        "worker_group": HashTranslator(),
        "user_id": HashTranslator(),
        "start_ts": IntTranslator(),
        "end_ts": IntTranslator(),
        "limits": FloatTranslator(),
        "cpu_time": IntTranslator(),
    }
    if duration_markers:
        res["duration"] = DurationsTranslator(duration_markers)
    return res


def is_slow(job, args):
    duration = float(job["duration"]) * args.slowdown
    return duration >= 60


def maybe_slow(job, args):
    ref = float(job["ref_duration"]) * args.slowdown
    return ref >= 60


def balance_data(data, args):
    slow = []
    maybe = []
    fast = []
    for job in data:
        if is_slow(job, args):
            slow.append(job)
        elif maybe_slow(job, args):
            maybe.append(job)
        else:
            fast.append(job)

    random.shuffle(maybe)
    random.shuffle(fast)
    res = slow + maybe[:len(slow)] + fast[:len(slow)]
    random.shuffle(res)
    return res


def get_queue_combinations(counts):
    if len(counts) == 0:
        return []
    if len(counts) == 1:
        return list(map(lambda x: [x], range(0, counts[0])))

    combinations = get_queue_combinations(counts[:-1])
    count = counts[-1]
    res = []
    for c in combinations:
        for i in range(0, count):
            combination = c.copy()
            combination.append(i)
            res.append(combination)
    return res


def normalize_combinations(combinations):
    combinations[:] = filter(lambda c: 2 not in c or 1 in c, combinations)
    for cmb in combinations:
        s = sum(cmb)
        if (s > 0):
            cmb[:] = map(lambda x: x / s, cmb)


def hot_one(indices):
    val = 1 / len(indices)
    res = [0.0, 0.0, 0.0, 0.0]
    for i in indices:
        res[i] = val
    return res


def compute_output(job, queues, args):
    emptyFast = [i for i in range(1, 4) if queues[i] == 0.0]
    if is_slow(job, args):  # slow job
        return hot_one([0])  # 0 = slow queue
    elif maybe_slow(job, args):  # potentialy slow job
        if queues[0] > 0.0 and len(emptyFast) > 0:
            return hot_one(emptyFast)  # slow is occupied and some fast queue is empty
        else:
            return hot_one([0])  # 0 = slow queue
    else:  # quick job
        if len(emptyFast) > 0:
            return hot_one(emptyFast)
        elif queues[0] == 0.0:
            return hot_one([0])  # 0 = slow queue
        else:
            # all queues are occupied -> shortest quick queue
            m = min(queues[1:])
            shortest = [i for i in range(1, 4) if queues[i] <= m]
            return hot_one(shortest)


def prepare_data(data, translators, inputs, args):
    queues = get_queue_combinations([2, 3, 3, 3])
    normalize_combinations(queues)

    output_data = []
    input_data = []
    progressBar = Bar('Preparing data', max=len(data))
    for job in data:
        for q in queues:
            input = list(map(
                lambda col: translators[col].translate(job[col]) if col in translators else job[col],
                inputs
            )) + q  # concatenate with queues configuration

            output = compute_output(job, q, args)
            input_data.append(input)
            output_data.append(output)

        progressBar.next()
    progressBar.finish()

    return input_data, output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Input CSV file path")
    parser.add_argument("hdf_file", type=str, help="Output hdf5 file path")
    parser.add_argument("--delimiter", default=";", type=str, help="CSV delimiter")
    parser.add_argument("--limit", default=1000000000, type=int, help="Maximal number of data loaded from CSV")
    parser.add_argument("--sus_threshold", default=2.0, type=float,
                        help="If duration exceeds limits multiplied by this threshold, the job is sus and ignored")
    parser.add_argument("--history", action="store_true", default=False,
                        help="Append another column with duration estimate from history")
    parser.add_argument("--inputs", default="tlgroup_id,exercise_id,runtime_id,worker_group,user_id,limits,cpu_time",
                        type=str, help="List of columns that will go into inputs dataset")
    parser.add_argument("--refs", default="", type=str, required=True,
                        help="CSV file with reference solutions (avg. ref solution time is appended to the list of columns)")
    parser.add_argument("--validation_size", default=0.1, type=float,
                        help="Relative size of the validation dataset (0.1 is 10%)")
    parser.add_argument("--slowdown", default=1.0, type=float,
                        help="Multiplicative constant for the duration.")
    args = parser.parse_args()

    translators = prepare_translators()
    inputs = args.inputs.split(",")
    if args.history:
        inputs.append("attempt")
        inputs.append("previous")

    print("Loading data ...")
    data = load_csv_data(args)
    if args.refs != "":
        inputs.append("ref_duration")
        load_reference_solutions(args, data)

    data = balance_data(data, args)

    split_point = len(data) - int(round(float(len(data)) * args.validation_size))
    training = data[:split_point]
    validation = data[split_point:]

    print("Writing {} ...".format(args.hdf_file))
    with h5py.File(args.hdf_file, "w") as f:
        training_input, training_output = prepare_data(training, translators, inputs, args)
        f.create_dataset("train-inputs", data=np.array(training_input, dtype='float32'))
        f.create_dataset("train-outputs", data=np.array(training_output, dtype='float32'))

        validation_input, validation_output = prepare_data(validation, translators, inputs, args)
        f.create_dataset("val-inputs", data=np.array(validation_input, dtype='float32'))
        f.create_dataset("val-outputs", data=np.array(validation_output, dtype='float32'))
