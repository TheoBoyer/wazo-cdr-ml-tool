import numpy as np
import pandas as pd
import os
import subprocess
import json
import argparse
import pickle
from flask_restplus import fields

def get_data_model(api):
    print(pipeline.get_input_headers())
    exit()
    return api.model('PickUp', {
        'name': fields.String,
        'class': fields.String(discriminator=True)
    })

def serve(args, models, pipeline):
    for c in columns:
        if c not in args:
            raise ValueError("{} missing. Required columns are {}".format(c, columns))
    entry = pd.DataFrame(args.to_dict(False))
    entry = pipeline.format_input(entry)
    entry = pipeline.forward(entry)
    predictions = []
    for m in models:
        predictions.append(m.predict(entry))

    prob = float(np.stack(predictions).mean(0)[0][0])
    return {
        "prob": prob,
        "label": get_label(prob)
    }

with open('./pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

columns = pipeline.requirements

data = pd.read_csv("validation_preds.csv")

preds = data["preds"].values
labels = data["labels"].values

sorted_args = np.argsort(preds)

sorted_preds = preds[sorted_args]
sorted_labels = labels[sorted_args]

zero_tresh_acc = 1 - (np.cumsum(sorted_labels) / (np.arange(len(sorted_labels)) + 1))
one_tresh_acc = np.flip(np.cumsum(np.flip(sorted_labels)) / (np.arange(len(sorted_labels)) + 1))

level_of_precision = 0.95
one_indic = one_tresh_acc >= level_of_precision
zero_indic = zero_tresh_acc >= level_of_precision

one_idx = np.min(np.arange(len(one_indic))[one_indic])
zero_idx = np.max(np.arange(len(zero_tresh_acc))[zero_indic])

zero_tresh = sorted_preds[zero_idx]
one_tresh = sorted_preds[one_idx]

def get_label(prob):
    label = "maybe"
    if prob >= one_tresh:
        label = "yes"
    elif prob <= zero_tresh:
        label = "no"
    return label

def main():
    parser = argparse.ArgumentParser(description='Give a probability and a text label to a context concerning wether or not the person is likely to pick up the call')
    parser.add_argument('-mute', action='store_true')
    args, _ = parser.parse_known_args()
    entry = {}
    for c in columns:
        v = input("" if args.mute else "{}: ".format(c))
        entry[c] = [v if len(v) else np.nan]
    entry = pd.DataFrame(entry)
    entry.to_csv("tmp", header=False, index=False)
    sub = subprocess.Popen("cat tmp | python main.py", shell=True, stdout=subprocess.PIPE)
    output = sub.stdout.readline()
    sub.stdout.close()
    sub.kill()
    if not args.mute:
        print()

    prob = float(json.loads(output)[0][0])
    label = get_label(prob)

    print(prob, label)
    os.remove("tmp")

if __name__ == "__main__":
    main()
