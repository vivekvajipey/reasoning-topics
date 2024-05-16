import torch
import pandas as pd
import os
import ast


df = None
pathname = "/Users/adityatadimeti/reasoning-topics/conditional/data/"
for path in os.listdir(pathname):
    if path.endswith(".csv"):
        if df is None:
            df = pd.read_csv(pathname + path)
        else:
            df = pd.concat([df, pd.read_csv(pathname + path)])

# Keep unique rows
df = df.drop_duplicates()


unique_questions = df["Question"].unique()
print("Number of questions", len(unique_questions)) # Number of questions


question_dfs = []
for question in unique_questions:
    question_dfs.append(df[df["Question"] == question])



for i in range(len(question_dfs)):
    set_of_buckets = set()
    for row in question_dfs[i].iterrows():
        replaced_row = row[1]["Merged buckets"].replace("[","").replace("]","")
        list_of_tuples = ast.literal_eval(f"[{replaced_row}]")
        second_elems = [x[1] for x in list_of_tuples]
        string_second_elems = "".join(second_elems)
        set_of_buckets.add(string_second_elems)
    print(f"Q{i} unique step orders: {len(set_of_buckets)}", f"Q{i} # generations {len(question_dfs[1])}")




for file_path in os.listdir("/Users/adityatadimeti/reasoning-topics/conditional/data/tensors/"):
    counter = 0
    if "CoT" not in file_path:
        continue
    if counter == 8:
        break

    dir = "/Users/adityatadimeti/reasoning-topics/conditional/data/tensors/" + file_path
    tensor = torch.load(dir, map_location="cpu") # Load the file into a tensor
    breakpoint()
    counter += 1

    # NOTE: BEST_MATCHES is a tuple, for each generation, that also contains the indices for starting and stopping of the steps in each generation. these can be used to get the entropy of each step and then plot it
    # along the edges