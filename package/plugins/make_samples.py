import os
import pickle
from distutils.dir_util import copy_tree

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

def main():
    folder = "examples/custom"
    if not os.path.isdir(folder):
        copy_tree("./examples", folder)

    for f_name in os.listdir(folder):
        with open(os.path.join(folder, f_name), 'r') as f:
            lines = f.readlines()
        if len(lines) == len(columns):
            sample = {k:v for k, v in zip(columns, lines)}
            with open(os.path.join(folder, f_name), 'w') as f:
                for c in other_columns:
                    f.write(sample[c])
