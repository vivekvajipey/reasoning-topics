"""
This file contains a toy version of the probability graph building algorithm.
"""
import numpy as np
from scipy.special import logsumexp
from queue import Queue
from collections import defaultdict, Counter
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import random
import openai
from openai import OpenAI
import json
from random import sample
import os
from dotenv import load_dotenv
from dotenv import load_dotenv
import re
from tqdm import tqdm

random.seed(0)
load_dotenv()  # Load variables from .env file

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = "You are a skilled mathematical reasoner who can identify the steps in an answer to a problem and provide concise, specific, and descriptive natural language labels for each step."
INITIAL_QUESTION_PROMPT = """I will give you a math problem, a solution to the problem, and individual steps. Categorize the steps using descriptive, specific, and concise natural language labels.   
You should return, in order of the problem's steps, the natural language labels you generated separated by commas in one single line. The goal is to turn your outputs into a graph by calling
output.split(", ") on the output you generate. 

Question: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?
Answer:
"""

BUCKET_ADDING_PROMPT = """I will give you a math problem, and a solution to the problem broken into steps. I will also give you a list of natural language label buckets that have been created from previous answers to the same 
question. Look at the solution's steps, and identify if each step fits into one of the existing buckets or if it should be in its own bucket.

You should return, in order of the problem's steps, the buckets each step belongs to, separated by commas in one single line. The goal is to turn your outputs into a graph by calling
output.split(", ") on the output you generate. These bucket names must be either the exact name or an existing bucket, or a new bucket preceded with the string "NEW :" The label name must be a descriptive, specific, and concise natural langauge. 
Do not generate a new bucket if a step fits into an existing bucket, even if the step is incorrect. 

Question: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?
"""

class GroundTruthModel:
    def __init__(self):
        self.states = [
            "question",
            "step 1, wording 1",
            "step 1, wording 2",
            "step 2, wording 1",
            "step 2, wording 2",
            "step 3, wording 1",
            "step 3, wording 2",
            "right answer, wording 1",
            "right answer, wording 2",
            "wrong answer",
        ]

    def get_next_state_distribution(self, path: tuple):
        """
        Define some toy dynamics over different reasoning traces.
        """
        unnormalized_probs = np.zeros(len(self.states))
        # if we've finished the problem, return an invalid distribution

        last_state_abstract = path[-1].split(",")[0] if "," in path[-1] else path[-1]
        if last_state_abstract in ("right answer", "wrong answer"):
            return unnormalized_probs

        unnormalized_probs[self.states.index("right answer, wording 1")] = 1
        unnormalized_probs[self.states.index("right answer, wording 2")] = 1
        unnormalized_probs[self.states.index("wrong answer")] = 2
        if last_state_abstract == "step 3":
            unnormalized_probs[self.states.index("right answer, wording 1")] = 5
            unnormalized_probs[self.states.index("right answer, wording 2")] = 5
        elif last_state_abstract in ("step 1", "step 2"):
            unnormalized_probs[self.states.index("step 3, wording 1")] = 1
            unnormalized_probs[self.states.index("step 3, wording 2")] = 10
        elif last_state_abstract == "question":
            unnormalized_probs[self.states.index("step 1, wording 1")] = 5
            unnormalized_probs[self.states.index("step 1, wording 2")] = 5
            unnormalized_probs[self.states.index("step 2, wording 1")] = 5
            unnormalized_probs[self.states.index("step 2, wording 2")] = 5

        probs = unnormalized_probs / np.sum(unnormalized_probs)
        return probs

    def sample(self, prompts):
        """
        Generate the next step given a partial path.
        """
        responses = []
        for prompt in prompts:
            next_state_dist = self.get_next_state_distribution(prompt)
            if sum(next_state_dist) == 0:
                responses.append("<eos>")
            else:
                next_state = np.random.choice(self.states, p=next_state_dist)
                responses.append(next_state)

        return responses

    def score(self, paths):
        """
        Get the log probs of each step along the paths.
        """
        scores = []
        for path in paths:
            score = 0
            for i in range(1, len(path)):
                score += np.log(
                    self.get_next_state_distribution(path[:i])[
                        self.states.index(path[i])
                    ]
                )
            scores.append(score)
        return scores


def is_paraphrase(s1, s2):
    """ """
    return s1 == s2 or s1.split(",")[0] == s2.split(",")[0]


def fetch_steps(mistral_answer):
    # Split using a regular expression that matches "<STEP " followed by one or more digits, and then ">"
    steps = re.split(r"<STEP \d+>", mistral_answer)
    # Filter out any empty strings resulting from the split
    return [step for step in steps if step]


# Can make this a generic "call gpt4" function later
def gpt4_create_buckets(prompt, max_tokens, temperature, mistral_answer_steps, retries = 5):
    while retries:
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        data = completion.choices[0].message.content.strip()
        list_output = data.split(", ")



        if len(list_output) == len(mistral_answer_steps):
            return list_output
        else:
            retries -= 1
            if retries == 0:
                print("Failed after multiple retries. Moving on..")
                return None

def gpt4_initialize_buckets(answers):
    answer_step_mapping = {}
    initialized_buckets = []
    for i, answer in tqdm(enumerate(answers)):
        question_number = i
        print(i, answer_step_mapping)
        mistral_answer_steps = fetch_steps(answer)
        mistral_answer_steps = [step.strip(":").strip() for step in mistral_answer_steps]
        if initialized_buckets == []:
            output = gpt4_create_buckets(f"{INITIAL_QUESTION_PROMPT} {mistral_answer_steps}", 1024, 0.7, mistral_answer_steps, 5)
            initialized_buckets = output
            answer_step_mapping[question_number] = output
        else:
            output = gpt4_create_buckets(f"{BUCKET_ADDING_PROMPT} \nBuckets: {initialized_buckets} \nAnswer broken into steps: {mistral_answer_steps}", 1024, 0.7, mistral_answer_steps, 5)
            if output == None:
                print("skipping", i)
                continue
            for i, step in enumerate(output):
                if "NEW" in step:
                    initialized_buckets.append(step.split(":")[-1].strip())
                    output[i] = step.split(":")[-1].strip()
            answer_step_mapping[question_number] = output

    return answer_step_mapping

def get_mistral_answers(pathname):
    with open(pathname, "r") as f:
        mistral_answers = f.read()
        answers = mistral_answers.split("<STEP 1>")
        answers = [("<STEP 1>" + answer) for answer in answers][1:]
        return answers


def construct_equivalence_classes(question : str, answers : list[str]):
    """
    Misc notes
    - last step (between <step split> and <end>) should be an "answer" step. Note that not all "answer" steps are the same; we should distinguish between incorrect answers.
    - Plan: "bucket initialization step" where we ask gpt4 to break down 10 individual answers to questions into buckets, then identify common buckets across them, then organize the steps into those buckets
    -        We hope that this is enough diversity to generate the initial buckets. Then, we sample repeatedly from Mistral, split the step, and ask if it fits under any of these steps or if it belongs in its own step. 
            repeat this process for all the samples

    the buckets that gpt4 generates need to be at the granularity of the steps that Mistral generates. This is why we should ask gpt4 to initialize the buckets given the steps already. 

    justification for the bucket initialization step: gpt4 can better scaffold initial buckets when given all the steps. There are tradeoffs to this approach: when given too many 
    paths and steps, gpt4 will tend to overgeneralize across buckets, and there's also context length limitations. Best of both worlds is to initialize the buckets, then ask gpt4 to categorize 
    steps as we go. 
    """

    # Bucket initialization step



def get_equivalence_classes(paths):
    """
    This function will be a lot more involved when working with LLMs
    

    
    <start>
    First, let's calculate the total distance Tim bikes during the workweek:
    Distance for one round trip to work = 2 * Distance to work
    = 2 * 20 miles
    = 40 miles per workday
    Total distance for 5 workdays = Distance per workday * Number of workdays
    = 40 miles * 5
    = 200 miles
    <step split>
    Next, add the weekend bike ride distance:
    Total weekly biking distance = Workweek distance + Weekend ride distance
    = 200 miles + 200 miles
    = 400 miles
    <step split>
    Now, let's calculate the total time he spends biking:
    Time = Total distance / Speed
    = 400 miles / 25 mph
    = 16 hours
    <step split>
    So, Tim spends a total of 16 hours biking in a week.
    <end>
    """

    equivalence_classes = defaultdict(Counter)
    for path in paths:
        last_state = path[-1]
        if last_state == "<eos>":
            continue
        last_state_abstract = (
            last_state.split(",")[0] if "," in last_state else last_state
        )
        equivalence_classes[last_state_abstract].update([path])

    return equivalence_classes


def sample_paths(path_counter, n_samples):
    """
    Sample from a path counter
    """
    return random.choices(
        list(path_counter.keys()), weights=list(path_counter.values()), k=n_samples
    )


def update_equivalence_classes(equivalence_classes, new_classes):
    """
    Update the set of equivalence classes with the newly-discovered ones
    """
    for new_class in new_classes:
        if new_class in equivalence_classes:
            equivalence_classes[new_class] += new_classes[new_class]
        else:
            equivalence_classes[new_class] = new_classes[new_class]

    return equivalence_classes



def get_all_transitions(model, n_samples=100):
    """
    Build a graph of the probability of each state given the previous state.
    """
    # keep track of the paths that end in each equivalence class
    all_equivalence_classes = {"question": Counter([("question",)])}
    # track the equivalence classes that we discovered in the last step
    new_equivalence_classes = list(all_equivalence_classes.keys())
    # track the different wordings we find that make up each equivalence class
    unique_wordings = defaultdict(set)
    unique_wordings["question"] = {"question"}

    while True:
        # reset the new equivalence classes
        prev_equivalence_classes = new_equivalence_classes
        new_equivalence_classes = []

        # sample paths from each existing equivalence class
        # (this should be batched if we're using a transformer)
        for class_name in prev_equivalence_classes:
            # sample some paths that lead to the new equivalence class
            paths = sample_paths(all_equivalence_classes[class_name], n_samples)

            for path in paths:
                # sample completions from the model to get the next step following the path
                completions = model.sample([path] * n_samples)
                new_paths = [path + (completion,) for completion in completions]

                # group the completions into equivalence classes
                completion_classes = get_equivalence_classes(new_paths)


                # Update our data structures
                # I think it's ok for this kind of thing to be in a for loop,
                # as each operation won't take much time
                for completion_class in completion_classes:
                    if completion_class not in all_equivalence_classes:
                        new_equivalence_classes.append(completion_class)
                    unique_wordings[completion_class].update(
                        [x[-1] for x in completion_classes[completion_class].keys()]
                    )
                # update the running tracker of all discovered euqivalence classes
                all_equivalence_classes = update_equivalence_classes(
                    all_equivalence_classes, completion_classes
                )

        # break when we stop discovering new equivalence classes
        if len(new_equivalence_classes) == 0:
            break

    return all_equivalence_classes, unique_wordings


def build_probabilistic_graph(classes, class_phrasings, model, n_samples):
    """
    Build a graph of the probability of transitioning from one state to another.
    """

    weighted_transitions = []

    # iterate over pairs of equivalence classes
    # NOTE: right now we're iterating over all possible classes. We could speed things up
    # by keeping track of the pairs of classes that were observed at least one in step 1
    for first_state in classes:
        for second_state in classes:
            # compute the probability of transitioning from first state to second state
            first_state_paths = sample_paths(classes[first_state], n_samples)

            # sum over phrasings of the second state.
            # This will most likely need to be batched
            all_logps_c2_given_s1 = []
            for s1_path in first_state_paths:
                # score the possible completions
                logps_s1_and_s2 = model.score(
                    [
                        s1_path + (s2_phrasing,)
                        for s2_phrasing in class_phrasings[second_state]
                    ]
                )
                # filter out -inf values
                logps_s1_and_s2 = np.array([p for p in logps_s1_and_s2 if p != -np.inf])
                if len(logps_s1_and_s2) == 0:
                    continue
                # subtract the probability of all but the last step to get the conditional probability
                logp_s1 = model.score([s1_path])[0]
                logps_s2_given_s1 = np.array(logps_s1_and_s2) - np.array(logp_s1)

                # sum over conditional probabilities for this equivalence class
                logp_c2_given_s1 = logsumexp(logps_s2_given_s1)
                all_logps_c2_given_s1.append(logp_c2_given_s1)

            # skip if the probability of going from first_state to second_state is 0
            if len(all_logps_c2_given_s1) == 0:
                continue

            # average over the samples to get the log prob estimate
            logp_c2_given_c1 = logsumexp(all_logps_c2_given_s1) - np.log(n_samples)

            weighted_transitions.append(
                (first_state, second_state, np.round(logp_c2_given_c1, 2))
            )

    # make a directed graph weighted by probabilities
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_transitions)
    return G


def create_graph():
    # read json file into dictionary from outputs_answer_mapping at time 2024-05-10_12-32-11.txt

    pairwise_probabilities = {}
    total_set = set()
    with open("outputs_answer_mapping at time 2024-05-10_12-32-11.txt", "r") as f:
        answer_mapping = json.loads(f.read())
        keys = list(answer_mapping.keys())[1: 5]  # Get all keys from the dictionary
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
        pos = nx.spring_layout(G)  # Position nodes with the spring layout

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges)
        plt.show()

        breakpoint()
if __name__ == "__main__":
    # model = GroundTruthModel()
    # classes, wordings = get_all_transitions(model)
    # graph = build_probabilistic_graph(classes, wordings, model, 100)

    # # graph drawing things
    # pos = nx.bfs_layout(graph, start="question")
    # labels = dict()
    # for u, v, data in graph.edges(data=True):
    #     labels[(u, v)] = data["weight"]
    # fig, ax = plt.subplots()
    # nx.draw_networkx(graph, pos=pos, ax=ax)
    # nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels)
    # fig.savefig("example-plot.png")


    # read from outputs_mistral_output at time 2024-05-09_22-54-34.txt
    # pathname = "outputs_mistral_output at time 2024-05-09_22-54-34.txt"
    # answers = get_mistral_answers(pathname)
    # answer_step_mapping = gpt4_initialize_buckets(answers)

        
    # # save answer_ampping to a file with the current time, and the word "answer_mapping" preceding it
    # from datetime import datetime
    # now = datetime.now()
    # current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    # current_time = "answer_mapping at time " + current_time
    # with open(f"outputs_{current_time}.txt", "w") as f:
    #     f.write(json.dumps(answer_step_mapping))
    
    create_graph()



    # mistral_answer = """First, lets calculate how much Tina earns for working a regular 8-hour shift: 
    # Regular hours earned = Hourly wage * Hours worked per day * Number of days 
    # Regular hours earned = $18.00 * 8 * 5 
    # Regular hours earned = $720.00 
    # <step split>
    # Next, lets calculate how many overtime hours she works in total: 
    # Overtime hours = Total hours worked - Regular hours 
    # Overtime hours = 10 hours * 5 days - 8 hours * 5 days 
    # Overtime hours = 10 * 5 - 8 * 5 Overtime hours = 50 hours - 40 hours 
    # Overtime hours = 10 hours 
    # <step split>
    # Now, lets calculate the overtime pay for those 10 hours: 
    # Overtime pay per hour = Hourly wage + 1/2 hourly wage 
    # Overtime pay per hour = $18.00 + $9.00 
    # Overtime pay per hour = $27.00 
    # Total overtime pay = Overtime pay per hour * Overtime hours 
    # Total overtime pay = $27.00 * 10 
    # Total overtime pay = $270.00 
    # <step split>
    # Finally, lets calculate her total earnings: 
    # Total earnings = Regular hours earned + Total overtime pay 
    # Total earnings = $720.00 + $270.00 
    # Total earnings = $990.00 
    # <step split>
    # So, Tina makes a total of $990.00 for working 10 hours every day for 5 days
    # <step split>"""

    # mistral_answer_steps = fetch_steps(mistral_answer)
    # output = gpt4_initialize_buckets(QUESTION_PROMPT + mistral_answer, 1024, 0.7, mistral_answer_steps, 5)
    # breakpoint()

    #def gpt4_initialize_buckets(prompt, max_tokens, temperature, mistral_answer_steps, retries = 5):

