import os
os.environ['HF_HOME'] = '/scr/vvajipey/.cache/huggingface'
os.environ['HF_HUB'] = '/scr/vvajipey/.cache/huggingface'
from dotenv import load_dotenv
load_dotenv(dotenv_path='../.env')
from huggingface_hub import login
login(os.getenv('HF_LOGIN_KEY'))

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
from collections import Counter, defaultdict
from difflib import get_close_matches
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from pprint import pprint
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F
from scipy.special import logsumexp
import networkx as nx
from accelerate import cpu_offload
import argparse
from utils import get_prompt_message
from torch import LongTensor, FloatTensor

class BatchSentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences_ids, device='cpu'):
        # Tokenize each stop sequence and store their token IDs
        # self.stop_token_ids_list = [tokenizer.encode(seq, add_special_tokens=False)[1:] for seq in stop_sequences] # "Step n:" type input
        self.stop_token_ids_list = [torch.tensor(stop_token_ids, device=device) for stop_token_ids in stop_sequences_ids] # ['. ', '\n', '. ']
        self.digit_tokens = [28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774]
        self.ignore_dups_indices = set([0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check each stop sequence against the end of the input_ids for each sequence in the batch
        for i, stop_token_ids in enumerate(self.stop_token_ids_list):
            if input_ids.shape[1] >= len(stop_token_ids):
                # Extract the last tokens of the same length as the stop sequence
                last_tokens = input_ids[:, -len(stop_token_ids):]
                # Check if they match the stop sequence tokens
                is_match = (last_tokens == stop_token_ids).all(dim=1)
                if is_match.any() and i in self.ignore_dups_indices:
                    second_to_last_tokens = input_ids[:, -2 * len(stop_token_ids): -len(stop_token_ids)]
                    if (second_to_last_tokens == stop_token_ids).all(dim=1):
                        continue 
                # If any sequence in the batch matches, return True to stop generation
                if is_match.any():
                    print("Stopping sequence detected, stopping generation.")
                    return True
        return False

def sample_paths(path_counter, n_samples):
    """
    Sample from a path counter
    """
    return random.choices(
        list(path_counter.keys()), weights=list(path_counter.values()), k=n_samples
    )

def convert_to_left_padded(tensor, model, tokenizer):
    trimmed_sequences = []
    for seq in tensor:
        # Find indices of the first and last non-pad tokens
        non_pad_indices = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            start_index = non_pad_indices[0]
            end_index = non_pad_indices[-1] + 1
            trimmed_sequences.append(seq[start_index:end_index])
        else:
            # if entire sequence is padding, use an empty sequence
            trimmed_sequences.append(torch.tensor([], dtype=torch.long, device=model.device))

    # Determine the maximum length after trimming
    max_length = max(len(seq) for seq in trimmed_sequences)
    padded_tensor = torch.full((tensor.shape[0], max_length), fill_value=tokenizer.pad_token_id, dtype=torch.long, device=model.device)

    # put left padding
    for i, seq in enumerate(trimmed_sequences):
        padded_tensor[i, -len(seq):] = seq

    return padded_tensor

def get_final_answer_numeric(step):
    prompt = f"""Directions:
Given a mathematical statement, I want you to extract only the numerical answer from the mathematical statement. You are provided with expert-annotated examples below. Do not output any text other than the number.
Input 1:
So, Tina makes a total of $1,179.00 in a week when she works 10 hours every day for 5 days.
Output 1:
1179
Input 2:
Therefore, Randy has a total of 60 mango trees + 60 coconut trees = 120 trees on his farm.
Output 2:
120
Input 3:
So it will take Joy 5 hours to read 120 pages.
Output 3:
5
Input 4:
Total time biking per week = 1.6 hours (work commute) + 8 hours (weekend ride) = 9.6 hours.
Output 4:
9.6
Input 5:
{step}
Output 5:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.0,
        n=1,
        stop=None
    )
    return response.choices[0].message.content 

def sample_completions_from_model(model, tokenizer, path, path_to_tensor, n_samples):
    if path[-1].startswith("<answer"):
        return None, path_to_tensor
    
    input_tensor = path_to_tensor[path]
    # print("OOOOOOOOOOOOOOOOOOOOOOOOO: ", input_tensor[-10:])

    if input_tensor[0, -1].item() == 2:
        answer_numeric = get_final_answer_numeric(path[-1])
        print("******ANSWER NUMERIC = ", answer_numeric)
        
        new_paths = [path + (f'<answer={answer_numeric}>',)] # no need to repeat n_samples times?
        print("NEW PATHS: ", new_paths)
        return new_paths, path_to_tensor

    # stop_sequences_ids = [[842, 28705], [13], [28723, 28705], [28723]]
    # stop_sequences_ids = [[842, 28705], [13], [28723, 28705]]
    # stop_sequences_ids = [[842, 28705], [28723, 28705]]
    # stop_sequences_ids = [[13]]
    # stop_sequences_ids = [[13], [28723, 28705], [28723], [842, 28705]]
    stop_sequences_ids = [[13], [842, 28705], [28723, 28705]]
    
    sentence_stopping_criteria = StoppingCriteriaList([BatchSentenceStoppingCriteria(tokenizer, stop_sequences_ids, device=model.device)])
    
    # max_length = 0
    new_paths = []

    for _ in range(n_samples):
        cur_input = input_tensor.to(model.device)
        MIN_STEP_LEN_TOKENS = 5
        MAX_RESAMPLE_ATTEMPTS = 5
        # print(f"WWWWWWWWW: {cur_input.shape[1]} - {input_tensor.shape[1]} > {MIN_STEP_LEN_TOKENS}")
        num_resamples = 0
        while (num_resamples < MAX_RESAMPLE_ATTEMPTS) and (cur_input.shape[1] - input_tensor.shape[1] <= MIN_STEP_LEN_TOKENS):
            gen_outputs = model.generate(
                input_tensor.to(model.device),
                min_new_tokens=10,
                max_new_tokens=1000,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                stopping_criteria=sentence_stopping_criteria
            ).sequences
            cur_input = gen_outputs

            num_resamples += 1

        # if gen_outputs.shape[1] > max_length:
            # max_length = gen_outputs.shape[1]

        completion = tokenizer.batch_decode(gen_outputs[:, input_tensor.shape[1]:])[0]

        PRINT_ALL_TOKENS = False
        if PRINT_ALL_TOKENS:
            for id in gen_outputs[0][input_tensor.shape[1]:]:
                print("id: ", id.item(), " -- tok: ", f"<{tokenizer.decode(id)}>")

        new_path = path + (completion,)
        new_paths.append(new_path)

        path_to_tensor[new_path] = gen_outputs

    return new_paths, path_to_tensor

def remove_trailing_step_info(step):
    # Regular expression to match "Step n:" at the end of the string, where n is any number
    pattern = r"\s*Step \d+:$"
    return re.sub(pattern, "", step).strip()

def get_bucket_prompt_simple(question, step, buckets=None, examples=None):
    if buckets is None:
        buckets = []
    if examples is None:
        examples = []
    
    bucket_examples_str = ""
    for i, (bucket, example) in enumerate(zip(buckets, examples), 1):
        bucket_examples_str += f"<BUCKET {i}>{bucket}</BUCKET {i}>\n<EXAMPLE from BUCKET {i}>{example}</EXAMPLE from BUCKET {i}>\n"
    
    return f"""I will give you a math problem, and a step in a solution to the problem. I will also give you a list of natural language label buckets that have been created from previous answers to the same question. Look at the step, identify which bucket it falls under, and return just the name of the bucket. If none of the existing buckets are representative, create a new bucket preceded with the string "NEW :". The label name must be descriptive, specific, and concise natural language. Return this new bucket string. 

Make sure that fundamentally different steps are put in different buckets; if two steps belong in the same bucket, ensure that the types of mathematical operations/fundamental logic are equivalent. Do not generate a new bucket if a step fits into an existing bucket. The bucket label should accurately summarize the contents of the step in enough detail to distinguish it from steps in other buckets.
If a step uses incorrect arithmetic, include "(Incorrect)" at the end of the bucket label. 
For example, "Total number of apples = 4 apples yesterday + 5 apples today = 9 apples total" would go into the "Adding total apples from yesterday and today" bucket, while "Total number of apples = 4 apples yesterday + 5 apples today = 11 apples total" and "Total number of apples = 4 apples yesterday + 5 apples today = 7 apples total" would go into the "Adding total apples from yesterday and today (Incorrect)" bucket. 
<QUESTION>{question}</QUESTION>
{bucket_examples_str}
<STEP TO CATEGORIZE>{step}</STEP TO CATEGORIZE>
Reminders: DO NOT INCLUDE tags such as <BUCKET> </BUCKET> or <STEP TO CATEGORIZE> </STEP TO CATEGORIZE>in your answer. If you are proposing a new bucket name, start with "NEW :" e.g. "NEW : bucket name"."""

def categorize_step_with_gpt4(question, step, eq_class_labels, eq_class_examples):
    prompt = get_bucket_prompt_simple(question, step, eq_class_labels, eq_class_examples)
    # print("+++PROMPT:\n", prompt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.0,
        n=1,
        stop=None
    )
    return response.choices[0].message.content

def get_equivalence_classes(paths, all_equivalence_classes, print_logging=False):
    equivalence_classes = defaultdict(Counter)
    current_eq_class_labels = [key for key in all_equivalence_classes.keys() if not key.startswith("question_")]
    
    # Loop over the current_eq_class_labels and sample one path for each label    
    current_eq_class_examples = []
    for key in current_eq_class_labels:
        prev_paths = list(all_equivalence_classes[key].keys())
        sampled_path = random.choice(prev_paths)
        current_eq_class_examples.append(sampled_path[-1])

    for path in paths:
        last_state = path[-1]
        if last_state.startswith("<answer"):
            equivalence_classes[last_state].update([path])
            continue
        # print("CURRENT PATH: ", path)
        # print("LAST STATE (TO CATEGORIZE): ", last_state)
        last_state_abstract = categorize_step_with_gpt4(path[0], last_state, current_eq_class_labels, current_eq_class_examples)
        if print_logging:
            print("+++++++++++++++++++++++++++++")
            print("gpt4o: ", last_state_abstract)#, "LAST STATE: ", last_state)
            print("+++++++++++++++++++++++++++++")

        if last_state_abstract.startswith("NEW : "):
            last_state_abstract = last_state_abstract[len("NEW : "):]
            current_eq_class_labels.append(last_state_abstract)
            current_eq_class_examples.append(path[-1])
            if print_logging:
                print("CREATING NEW BUCKET: ", last_state_abstract)
        else:
            # Find the closest string match between the response and current_eq_class_labels
            closest_match = get_close_matches(last_state_abstract, current_eq_class_labels, n=1, cutoff=0.75)
            if closest_match:
                if print_logging:
                    print(f"ADDING '{last_state_abstract}' TO EXISTING BUCKET", closest_match[0])
                last_state_abstract = closest_match[0]
            else:
                current_eq_class_labels.append(last_state_abstract)
                current_eq_class_examples.append(path[-1])
                if print_logging:
                    print("NO CLOSEST MATCH, NEW BUCKET: ", last_state_abstract)
        
        equivalence_classes[last_state_abstract].update([path])

    return equivalence_classes

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

def get_all_transitions(model, question_index, gsm_df, n_samples=2, print_logging=False):
    """
    Build a graph of the probability of each state given the previous state.
    """
    question = gsm_df['question'].tolist()[question_index]
    # path_to_tensor = {(question,) : tokenizer.apply_chat_template(prompt_with_step_splits_numbered(question), add_generation_prompt=True, return_tensors="pt")}
    # print(get_prompt_message(question, num_fewshot_examples=0))
    path_to_tensor = {(question,) : tokenizer.apply_chat_template(get_prompt_message(question, num_fewshot_examples=0), add_generation_prompt=True, return_tensors="pt")}

    question_abstract = f"question_{question_index}"
    # keep track of the paths that end in each equivalence class
    all_equivalence_classes = {question_abstract : Counter([(question,)])}
    # track the equivalence classes that we discovered in the last step
    new_equivalence_classes = list(all_equivalence_classes.keys())
    # track the different wordings we find that make up each equivalence class
    unique_wordings = defaultdict(set)
    unique_wordings[question_abstract] = {question}

    next_eq_classes = defaultdict(set) # eq_class -> set of next step equivalence classes 

    iteration = 0
    while True:
        print(f"Iteration {iteration}: {len(new_equivalence_classes)} branches")

        if print_logging:
            print("CURRENT ALL_EQUIVALENCE_CLASSES: ", all_equivalence_classes)
            print("MOST RECENT EQUIVALENCE CLASSES: ", new_equivalence_classes)
        # reset the new equivalence classes
        prev_equivalence_classes = new_equivalence_classes
        new_equivalence_classes = []

        # sample paths from each existing equivalence class
        # (this should be batched if we're using a transformer)
        for class_name in prev_equivalence_classes:
            # if print_logging:
            print()
            print("->class name: ", class_name)
                # print()
            
            num_paths_to_sample = n_samples
            num_completions_to_sample = n_samples
            if iteration > 2:
                # sample only min(number of traces ending in class_name, n_samples)
                # do not want to over sample for rare states
                num_paths_to_sample = min(n_samples, sum(all_equivalence_classes[class_name].values()))
                if "Incorrect" in class_name or "incorrect" in class_name:
                    print("ONLY SAMPLING FROM INCORRECT ONCE")
                    num_completions_to_sample = min(1, n_samples)


            # sample some paths that lead to the new equivalence class
            paths = sample_paths(all_equivalence_classes[class_name], num_paths_to_sample)
            if print_logging:
                print("paths: ", paths)
                print()

            for i, path in enumerate(paths):
                if i == 0:
                    print("example: ", path[-1])

                # sample completions from the model to get the next step following the path
                new_paths, path_to_tensor = sample_completions_from_model(model, tokenizer, path, path_to_tensor, n_samples=num_completions_to_sample)

                # if terminal cur path ends in terminal state ("<answer=...>"), continue
                if new_paths == None:
                    continue

                if print_logging:
                    print("new_paths: ", new_paths)

                # group the completions into equivalence classes
                completion_classes = get_equivalence_classes(new_paths, all_equivalence_classes)
                if print_logging:
                    print("completions_classes: ", completion_classes)
                
                for completion_class in completion_classes.keys():
                    next_eq_classes[class_name].add(completion_class) 

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
        if print_logging:
            print("LEN(NEW_EQ_CLASSES): ", len(new_equivalence_classes))
            print("=====================================================")
            print()

        print("Current state of the tree of solutions after Iteration ", iteration, ":")
        eqc_idx = 0
        for eq_class, cntr in all_equivalence_classes.items():
            print(f"Equivalence Class {eqc_idx}: {eq_class}, Number of Paths: {len(cntr)}, Path Lengths: {[len(trace) for trace in list(cntr.keys())]}")
            eqc_idx += 1
        iteration += 1
        print()

        if len(new_equivalence_classes) == 0:
            break

    return all_equivalence_classes, unique_wordings, path_to_tensor, next_eq_classes


# BUILD PROBABILISTIC GRAPH
def append_s2_to_s1_path_tensor(s1_path_tensor, s2_phrasings, tokenizer, question_length):
    """
    Stack s1_path_tensor with each s2_phrasing tensor and create corresponding masks.
    
    Parameters:
    - s1_path_tensor: Tensor of the s1 path.
    - s2_phrasings: List of s2 phrasing strings.
    - tokenizer: The tokenizer used for encoding.
    - question_length: Length of the question portion in s1_path_tensor.
    
    Returns:
    - stacked_tensor: Tensor containing stacked s1_path and s2_phrasing tensors.
    - stacked_masks: Mask tensor with 0s for the question portion and padding, 1s for the rest.
    """
    prefix = s1_path_tensor.view(-1)
    suffixes = [
            tokenizer.encode(s2_phrasing, add_special_tokens=False, return_tensors='pt').view(-1)
            for s2_phrasing in s2_phrasings
        ]
    num_suffixes = len(suffixes)
    max_length = prefix.shape[0] + max(s.shape[0] for s in suffixes)

    new_tensor = torch.zeros((num_suffixes, max_length), dtype=prefix.dtype)
    new_mask = torch.zeros((num_suffixes, max_length), dtype=torch.uint8)

    for i in range(num_suffixes):
        # print("devs (p, s[i]): ", prefix.device, suffixes[i].device)
        combined_trace = torch.cat((prefix.cpu(), suffixes[i]))
        new_tensor[i, :combined_trace.size(0)] = combined_trace

        # mask is 0 for question tokens, 1 for reasoning step tokens
        new_mask[i, question_length : combined_trace.size(0)] = 1 # might include <eos> token in logprobs
    
    return new_tensor, new_mask

def score_paths_with_base_model(model, paths_tensor, mask, print_all_logprobs = False, tokenizer=None):
    # paths tensor is 2D matrix with each row being a path
    paths_tensor = paths_tensor.to(model.device)
    mask = mask.to(model.device) 

    with torch.no_grad():
        outputs = model(paths_tensor)
    
    logprobs = torch.log_softmax(outputs.logits, dim=1).detach()
    logprobs = logprobs[:, :-1, :]
    paths_tensor = paths_tensor[:, 1:]

    gen_logprobs = torch.gather(logprobs, 2, paths_tensor[:, :, None]).squeeze(-1)
    masked_logprobs = gen_logprobs * mask[:, 1:].float() 
    summed_logprobs = masked_logprobs.sum(dim=1)

    if print_all_logprobs:
            print("summed_probs: ", summed_logprobs.shape)
            for input_sentence, input_probs in zip(paths_tensor , masked_logprobs):
            # for input_sentence, input_probs in zip(s1_s2_tensor , gen_logprobs): # check all logprobs
                for token, p in zip(input_sentence, input_probs):
                    if token not in tokenizer.all_special_ids:
                        print(f"{tokenizer.decode(token)} ({token}): {p.item()}")

    return summed_logprobs

def build_probabilistic_graph(classes, class_phrasings, path_to_tensor, next_eq_classes, base_model, tokenizer, n_samples, question, print_logging=False, file_name="unnamed"):
    """
    Build a graph of the probability of transitioning from one state to another.
    """

    weighted_transitions = []

    q_tensor = path_to_tensor[(question,)]
    question_length = q_tensor.size(1)

    # iterate over pairs of equivalence classes
    # NOTE: right now we're iterating over all possible classes. We could speed things up
    # by keeping track of the pairs of classes that were observed at least one in step 1
    for first_state in classes:
        for second_state in next_eq_classes[first_state]:
            print("first state: ", first_state, ", second state: ", second_state)
            # compute the probability of transitioning from first state to second state
            first_state_paths = sample_paths(classes[first_state], n_samples)

            s2_phrasings = class_phrasings[second_state]

            # sum over phrasings of the second state.
            # This will most likely need to be batched
            all_logps_c2_given_s1 = []
            for s1_path in first_state_paths:
                # score the possible completions
                s1_path_tensor = path_to_tensor[s1_path]
                s1_s2_tensor, s1_s2_mask = append_s2_to_s1_path_tensor(s1_path_tensor, s2_phrasings, tokenizer, question_length)
                logps_s1_and_s2 = score_paths_with_base_model(base_model, s1_s2_tensor, s1_s2_mask, print_all_logprobs=False).cpu()
                # filter out -inf values
                logps_s1_and_s2 = np.array([p for p in logps_s1_and_s2 if p != -np.inf])
                if len(logps_s1_and_s2) == 0:
                    continue
                # subtract the probability of all but the last step to get the conditional probability
                s1_no_q_mask = torch.ones_like(s1_path_tensor)
                s1_no_q_mask[:, :question_length] = 0

                logp_s1 = score_paths_with_base_model(base_model, s1_path_tensor, s1_no_q_mask, print_all_logprobs = False).cpu()
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

    # Define the file path to save the weighted transitions
    weighted_transitions_file_path = f'weighted_transitions/weighted_transitions_{file_name}.pkl'

    # Save the weighted transitions to a pickle file
    with open(weighted_transitions_file_path, 'wb') as f:
        pickle.dump(weighted_transitions, f)

    # make a directed graph weighted by probabilities
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_transitions)
    return G

if __name__ == "__main__":
    gsm_df = pd.read_csv('../distribution/data/gsm8kTest.csv')
    # PARAMETER SELECTION
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--q_num', type=int, default=0, help='Index of the question to process')
    parser.add_argument('--n_samples', type=int, default=2, help='Number of samples to generate')
    parser.add_argument('--checkpoint_load', action='store_true', help='Flag to load from checkpoint')
    model_name = "mistral-7b-v0.1"

    args = parser.parse_args()

    question_index = args.q_num
    question = gsm_df['question'].tolist()[question_index]
    print("Question:")
    print(question)
    n_samples = args.n_samples
    checkpoint_load = args.checkpoint_load

    file_id = f"n{n_samples}p{question_index}"

    if not checkpoint_load:
        name2instruct = {"mistral-7b-v0.1":"mistralai/Mistral-7B-Instruct-v0.1"}
        instruct_model_name = instruct_model_name = name2instruct[model_name]
        tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="auto")
        instruct_model.generation_config = GenerationConfig.from_pretrained(instruct_model_name)
        instruct_model.generation_config.pad_token_id = instruct_model.generation_config.eos_token_id

        all_equivalence_classes, unique_wordings, path_to_tensor, next_eq_classes = get_all_transitions(instruct_model, question_index, gsm_df, n_samples=n_samples, print_logging=False)

        os.makedirs('pickle_jar', exist_ok=True)
        with open(f'pickle_jar/all_equivalence_classes_{file_id}.pkl', 'wb') as f:
            pickle.dump(all_equivalence_classes, f)

        with open(f'pickle_jar/unique_wordings_{file_id}.pkl', 'wb') as f:
            pickle.dump(unique_wordings, f)
        
        with open(f'pickle_jar/next_eq_classes_{file_id}.pkl', 'wb') as f:
            pickle.dump(next_eq_classes, f)

        # Save path_to_tensor using torch's save method
        torch.save(path_to_tensor, f'pickle_jar/path_to_tensor_{file_id}.pt')
        cpu_offload(instruct_model)

    # CHECKPOINT LOAD
    if checkpoint_load:
        print("Loading all_equivalence_classes, unique_wordings, path_to_tensor from checkpoint...")
        with open(f'pickle_jar/all_equivalence_classes_{file_id}.pkl', 'rb') as f:
            all_equivalence_classes = pickle.load(f)

        with open(f'pickle_jar/unique_wordings_{file_id}.pkl', 'rb') as f:
            unique_wordings = pickle.load(f)

        path_to_tensor = torch.load(f'pickle_jar/path_to_tensor_{file_id}.pt')
    
    # build prob graph
    name2base = {"mistral-7b-v0.1":"mistralai/Mistral-7B-v0.1"}

    base_model_name = name2base[model_name]
    print("Loading ", base_model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.generation_config = GenerationConfig.from_pretrained(base_model_name)
    base_model.generation_config.pad_token_id =base_model.generation_config.eos_token_id

    graph = build_probabilistic_graph(all_equivalence_classes, unique_wordings, path_to_tensor, next_eq_classes, base_model, base_tokenizer, n_samples, question, file_name=file_id)
    pos = nx.bfs_layout(graph, start=f"question_{question_index}", scale=100)
    labels = dict()
    for u, v, data in graph.edges(data=True):
        labels[(u, v)] = data["weight"]
    fig, ax = plt.subplots()
    nx.draw_networkx(graph, pos=pos, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels)
    fig.savefig(f"sentence_figs/{file_id}-plot.png") 