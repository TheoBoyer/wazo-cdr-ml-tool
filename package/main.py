import pickle
import os
import json
import pandas as pd
import numpy as np
import argparse
import time

from core.utils import load_dataset, load_model_class
from datasets.CDR import DataPipeline

with open("./config.json") as f:
    config = json.load(f)

model_classes = {}
for p in config["packages"]:
    if p['attribute'] is None:
        DataPipeline = load_dataset(p['name'])
    else:
        model_classes[p['name']] = load_model_class(p['name'])

model_dir = "./trained_models"

models = []
for model_file in sorted(os.listdir(model_dir)):
    models.append(model_classes[config['model_type']].load(os.path.join(model_dir, model_file)))

with open('./pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

headers = pipeline.get_input_headers()

def parse_item(v, t):
    if v == "":
        return None
    if t == "int64":
        return int(v)
    if t == "bool":
        return v == 'True'
    if t == "float64":
        return float(v)
    if t == "object":
        return v
    raise ValueError("Unknown type: {}".format(t))

def try_parse(line, header):
        if len(line) != len(header):
            return None
        data = {}
        for h, v in zip(header.items(), line):
            try:
                data[h[0]] = parse_item(v, h[1])
            except ValueError:
                return None
        return data

def parse_input():
    done = False
    data = []
    while not done:
        try:
            line = input()
            if line == "":
                done = True
                break
            line = line.split(',')
            for h in headers:
                data_point = try_parse(line, h)
                if data_point is not None:
                    data.append(data_point)
                    break
            else:
                raise ValueError("No header matched. possible options are: {}".format(headers))
            
        except EOFError:
            done = True
    if len(data):
        data = pd.DataFrame(data)
        data = pipeline.format_input(data)
        X = pipeline.forward(data)
        return X
    return None

def serve(endless=True, time_inference=False):
    while True:
        input = parse_input()
        if input is not None:
            tstart = time.time()
            predictions = []
            for m in models:
                predictions.append(m.predict(input))

            predictions = np.stack(predictions).mean(0)
            if time_inference:
                print("Inference time: {:.4f}s".format(time.time()-tstart))
            print(predictions.tolist())
        if not endless:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serve a machine learning model. Accepted headers are: {}'.format(headers))
    parser.add_argument('-endless', action='store_true')
    parser.add_argument('-time_inference', action='store_true')
    args, _ = parser.parse_known_args()

    serve(args.endless, args.time_inference)
