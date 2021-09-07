import numpy as np
import pandas as pd
import os
import subprocess
import ast
import pickle
import argparse

n_min=20

def serve(args, models, pipeline):
    for c in other_columns:
        if c not in args:
            raise ValueError("{} missing. Required columns are {}".format(c, other_columns))
    entry = make_entry_list(args.to_dict())
    entry = pipeline.format_input(entry)
    entry = pipeline.forward(entry)
    print(entry)
    predictions = []
    for m in models:
        predictions.append(m.predict(entry))

    predictions = np.stack(predictions).mean(0)
    entry["preds"] = predictions.tolist()
    entry = entry.explode('preds')
    entry["preds"] = entry["preds"].astype(float)
    result = entry[["requested_extension", "preds"]].groupby(['requested_extension']).aggregate({"preds": 'mean'}).sort_values("preds", ascending=False)
    result = result[~(result.index.str.endswith('nan'))]
    result["requested_extension"] = result.index
    print(result)
    return result.to_dict('records')

with open('./pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

columns = pipeline.requirements

dest_columns = []
other_columns = []
for c in columns:
    if "destination" in c or "requested" in c:
        dest_columns.append(c)
    else:
        other_columns.append(c)

data = pd.read_csv("validation_preds.csv")

def make_entry_list(entry):
    global n_min
    print(entry)
    identities = data[dest_columns].groupby(dest_columns).size().reset_index()#.drop(0, axis=1)
    identities = identities[identities[0] > n_min].drop(0, axis=1)
    for c in dest_columns:
        identities = identities[~(identities[c].str.contains('rare'))]
        identities = identities[identities[c].str.contains('stack' + entry['stack'])]

    df = {k: [] for k in entry.keys()}
    for k, v in entry.items():
        df[k] = [v] * len(identities)

    df = pd.DataFrame(df).join(identities.reset_index())
    return df[columns]

def main():
    global n_min
    parser = argparse.ArgumentParser(description='Give a probability and a text label to a context concerning wether or not the person is likely to pick up the call')
    parser.add_argument('-mute', action='store_true')
    parser.add_argument("--min_count", type=int, default=20, help="Minimum number of occurences of an unique ensemble of destination to be taken into account")
    args, _ = parser.parse_known_args()
    n_min = args.min_count
    entry = {}
    for c in other_columns:
        v = input("" if args.mute else "{}: ".format(c))
        entry[c] = v if len(v) else np.nan

    entry = make_entry_list(entry)
    entry.to_csv("tmp", header=False, index=False)
    sub = subprocess.Popen("cat tmp | python main.py", shell=True, stdout=subprocess.PIPE)
    output = "".join(sub.stdout.read().decode('utf-8').split("\n"))
    sub.stdout.close()
    sub.kill()
    output = ast.literal_eval(output)
    entry["preds"] = output
    entry = entry.explode('preds')
    entry["preds"] = entry["preds"].astype(float)
    result = entry[["requested_extension", "preds"]].groupby(['requested_extension']).aggregate({"preds": 'mean'}).sort_values("preds", ascending=False)
    result = result[~(result.index.str.endswith('nan'))]
    print(result)
    os.remove("tmp")

if __name__ == "__main__":
    main()
