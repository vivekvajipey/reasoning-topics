"""
Empirical graph building script.
"""
import os
import openai
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy.special import logsumexp
from queue import Queue
from collections import defaultdict, Counter
from itertools import product
import networkx as nx
import random

import json
from random import sample

import matplotlib.pyplot as plt

import re
from tqdm import tqdm
from tensor_processing import get_stripped_outputs

random.seed(0)

QUESTION = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"
SYSTEM_PROMPT = "You are a skilled mathematical reasoner who can identify the steps in an answer to a problem and provide concise, specific, and descriptive natural language labels for each step."
INITIAL_QUESTION_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Categorize the solution into steps using descriptive, specific, and concise natural language labels.   
You should return the steps, in order, separated by commas in one single line. The goal is to call output.split(", ") on the output you generate. 

Question: {QUESTION}
Answer:
"""

BUCKET_ADDING_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. I will also give you a list of natural language buckets that have been created to categorize steps from previous answers to the same question.
Look at the solution, break it into steps, and identify which bucket it fits into. If none of the buckets are representative, create a new bucket preceded with the string "NEW :" which is a descriptive, specific, and concise natural language label.
Do not generate a new bucket if a step fits into an existing bucket, even if the step is incorrect. 

You should return the buckets, in order of the problem's steps, separated by commas in one single line. The goal is to call output.split(", ") on the output you generate. 

Question: {QUESTION}
"""


# Can make this a generic "call gpt4" function later 
def gpt4_create_buckets(prompt, temperature, retries = 5): # using default max_tokens
    counter = retries
    while counter:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        data = completion.choices[0].message.content.strip()
        list_output = data.split(", ")

        if len(list_output) > 1: # GPT-4o actually generated steps
            return list_output
        else:
            counter -= 1
            print(f"Retrying with {counter} retries left.")
            if counter == 0:
                print(f"Failed after {retries} retries. Moving on.")
                return None

def gpt4_initialize_buckets(answers):
    answer_step_mapping = {}
    initialized_buckets = []
    for i, answer in tqdm(enumerate(answers)):
        question_number = i
        print(i, answer_step_mapping)

        if initialized_buckets == []:
            output = gpt4_create_buckets(INITIAL_QUESTION_PROMPT, 0, 5) # using temperature 0 to get consistent buckets 
            initialized_buckets = output
            answer_step_mapping[question_number] = output
        else:
            output = gpt4_create_buckets(f"{BUCKET_ADDING_PROMPT} \nBuckets: {initialized_buckets} \nAnswer: {answer}", 0, 5)
            if output == None:
                print("Skipped", i)
                continue
            for i, step in enumerate(output):
                if "NEW" in step:
                    initialized_buckets.append(step.split(":")[-1].strip())
                    output[i] = step.split(":")[-1].strip()
            answer_step_mapping[question_number] = output
    return answer_step_mapping

def create_graph():
    # read json file into dictionary from outputs_answer_mapping at time 2024-05-10_12-32-11.txt

    pairwise_probabilities = {}
    total_set = set()
    with open("outputs_answer_mapping at time 2024-05-10_12-32-11.txt", "r") as f:
        answer_mapping = json.loads(f.read())
        keys = list(answer_mapping.keys())[1: 2]  # Get all keys from the dictionary
        # Create a new dictionary with these selected entries
        answer_mapping = {key: answer_mapping[key] for key in keys}
       
        for key in answer_mapping:
            for value in answer_mapping[key]:
                total_set.add(value)
        
        for entry in total_set:
            pairwise_probabilities[entry] = {}
            pairwise_probabilities["start"] = {}
            pairwise_probabilities[entry]["end"] = 0
            for entry2 in total_set:
                pairwise_probabilities[entry][entry2] = 0
                

        for key in answer_mapping:
            for i in range(0, len(answer_mapping[key])):
                if i == 0:
                    pairwise_probabilities["start"][answer_mapping[key][i]] = 1
                elif i == len(answer_mapping[key]) - 1:
                    pairwise_probabilities[answer_mapping[key][i]]["end"] = 1
                    pairwise_probabilities[answer_mapping[key][i - 1]][answer_mapping[key][i]] += 1
                else:
                    pairwise_probabilities[answer_mapping[key][i - 1]][answer_mapping[key][i]] += 1
        
        # Normalize the probabilities
        for i in pairwise_probabilities.keys():
            total = sum(pairwise_probabilities[i].values())
            for j in pairwise_probabilities[i].keys():
                if total != 0:
                    pairwise_probabilities[i][j] /= total
                    pairwise_probabilities[i][j] = round(pairwise_probabilities[i][j], 2)

        # Create a list of names for the nodes
        names = list(total_set)

        # Generate a random adjacency matrix
       #m = np.random.random((N, N))
        m = pairwise_probabilities
        #np.fill_diagonal(m, 0)  # No loops, set diagonal to 0

        # Create edges with labels based on the random adjacency matrix
        
        edges = {}
        for i in pairwise_probabilities.keys():
            for j in pairwise_probabilities[i].keys():
                if pairwise_probabilities[i][j] != 0:
                    edges[(i, j)] = pairwise_probabilities[i][j]

        # Create a directed graph, add nodes with names, and add labeled edges
        G = nx.DiGraph()
        G.add_nodes_from(names)
        G.add_edges_from(edges.keys())
        #pos = nx.spring_layout(G)  # Position nodes with the spring layout
        fixed_positions = {"start": (0, 2.5), "end": (30, 2.5)}
        
        pos = nx.bfs_layout(G, start = "start")

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges)
       # plt.figure(figsize=(10, 5))  # Adjust the size as needed
        plt.show()


        breakpoint()


if __name__ == "__main__":
    file_path = "/Users/adityatadimeti/reasoning-topics/conditional/data/mistral-7b-v0.1-samples10-fewshot0-temp0.7-topk40-CoT-gsm8k_p9-1.pt"
    question_start = "[INST] Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\nA: Let's think step by step. [/INST]"
    mistral_answers = get_stripped_outputs(file_path=file_path, question_start=question_start) 
    breakpoint()

    # read from outputs_mistral_output at time 2024-05-09_22-54-34.txt
    pathname = "outputs_mistral_output at time 2024-05-09_22-54-34.txt"
    answers = get_mistral_answers(pathname)
    answer_step_mapping = gpt4_initialize_buckets(answers)

        
    # save answer_ampping to a file with the current time, and the word "answer_mapping" preceding it
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    current_time = "answer_mapping at time " + current_time
    with open(f"outputs_{current_time}.txt", "w") as f:
        f.write(json.dumps(answer_step_mapping))
    
    create_graph()


