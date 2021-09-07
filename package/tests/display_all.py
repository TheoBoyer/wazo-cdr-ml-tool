import subprocess
import os
import pickle

plugin_template = "cat {} | python run_plugin.py {} -mute"
whoshouldicall_file_template = "examples/custom/sample{}.txt"
willyoupickup_file_template = "examples/sample{}.txt"

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

def run_cmd(cmd_str):
    sub = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
    output = sub.stdout.read().decode('utf-8')
    sub.stdout.close()
    sub.kill()
    return output

def run():
    run_cmd("python run_plugin.py make_samples")
    i = 0
    while os.path.exists(willyoupickup_file_template.format(i)):
        print()
        print("#"*80)
        print("#"*10, "RUNNING \"willyoupickup\" ON", willyoupickup_file_template.format(i), "#"*10)
        print("#"*80)
        print()
        with open(willyoupickup_file_template.format(i), 'r') as f:
            lines = f.readlines()
        print('## Input:')
        for k, v in zip(columns, lines):
            print("{}:\t{}".format(k, v.split("\n")[0]))
        print()
        output = run_cmd(plugin_template.format(willyoupickup_file_template.format(i), "willyoupickup"))
        print('## Output:')
        print(output)

        print()
        print("#"*80)
        print("#"*10, "RUNNING \"whoshouldicall\" ON", whoshouldicall_file_template.format(i), "#"*10)
        print("#"*80)
        print()
        with open(willyoupickup_file_template.format(i), 'r') as f:
            lines = f.readlines()
        print('## Input:')
        for k, v in zip(other_columns, lines):
            print("{}:\t{}".format(k, v.split("\n")[0]))
        print()
        output = run_cmd(plugin_template.format(whoshouldicall_file_template.format(i), "whoshouldicall"))
        print('## Output:')
        print(output)
        
        i += 1
    