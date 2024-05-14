import torch
from transformers import AutoTokenizer
import os
import openai
from openai import OpenAI
from difflib import SequenceMatcher
import random
import datetime

# Function to calculate similarity
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

QUESTION_START = "[INST] Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\nA: Let's think step by step. [/INST]"
FILE_PATH = "/Users/adityatadimeti/reasoning-topics/conditional/data/mistral-7b-v0.1-samples10-fewshot0-temp0.7-topk40-CoT-gsm8k_p9-1.pt"
SYSTEM_PROMPT = "You are a skilled mathematical reasoner."
QUESTION = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"

INITIAL_QUESTION_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Categorize the solution into steps using descriptive, specific, and concise natural language labels.   
You should return the steps, in order, separated by commas in one single line. The goal is to call output.split(", ") on the output you generate. 

Question: {QUESTION}
Answer:"""

TEMP_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Your task is to identify where in the solution distinct steps occur, and to separate the solution at the location of each step with <STEP SPLIT>.
You should return the steps, in order, separated by <STEP SPLIT> in one single line. The goal is to call output.split("<STEP SPLIT>") on the output you generate. 
If I concatenate the steps you provide, they should be identical to the original solution.

Question: {QUESTION}
Answer:
"""

BUCKET_ASSIGNING_PROMPT = f"""I will give you a math problem, a model-generated solution to the problem, and a substep from the model-generated solution. You should return a descriptive, specific, and concise
natural language label for the substep.

Question: {QUESTION}
Answer:"""

MERGE_BUCKETS_PROMPT = f"""Based on this question: {QUESTION}, would you say these 2 model-generated names of solutions steps refer to the same thing? Return just a "Yes" or "No"
"""

MERGE_BUCKETS_WITH_STEPS_PROMPT = f"""Based on this question: {QUESTION}, would you say these 2 substeps of model-generated solutions perform the same steps and should be classified under the same bucket? Return just a 'Yes' or 'No' """

# new prompt to try that i havent yet MERGE_BUCKETS_WITH_STEPS_PROMPT = f"""Based on this question: {QUESTION}, does this new model-generated substep belong in this existing list of substeps that have been categorized under the same bucket? Return just a 'Yes' or 'No' """


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

tensor = torch.load(FILE_PATH, map_location="cpu") # Load the file into a tensor
decoded_output = tokenizer.batch_decode(tensor)
response = " ".join(decoded_output)
parsed_outputs = response.split(QUESTION_START)[1:]


stripped_outputs = [output.replace("<s>", "").replace("</s>", "").strip() for output in parsed_outputs] # Remove <s> tokens from the list of outputs


buckets = []
merged_buckets = []

bucket_step_mapping = {} # maps a defined bucket to all the steps that have been assigned to it.
answer_bucket_mapping = {} # maps a defined answer to a list of buckets that have been identified, post per-answer merging. 
random.seed(0)
for i in range(tensor.shape[0]):
    print(i, len(bucket_step_mapping))
    #breakpoint()
    completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{TEMP_PROMPT} {stripped_outputs[i]}"}
            ],
            temperature = 0
        )
    data = completion.choices[0].message.content.strip()
    list_output = data.split("<STEP SPLIT>")
    best_matches = []
    #assert("".join(list_output) == stripped_outputs[i]) # Since the steps are separated by <STEP SPLIT>, we have to assert that joining them is equal to the original output. (NOTE THIS IS NOT ALWAYS TRUE SO I COMMENTED IT OUT)

    for step_var in list_output:
        token_length = len(tokenizer.encode(step_var)) - 1 # subtract 1 for the <s> token
        
        best_similarity = 0
        best_index = -1

        found = False
        for start_index in range(1, len(tensor[i]) - token_length):
            decoded_segment = tokenizer.decode(tensor[i][start_index : start_index + token_length])
            similarity = calculate_similarity(decoded_segment, step_var)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = start_index
            
        if best_index != -1:
            best_matches.append((step_var, best_index, best_index + token_length, best_similarity))

    # for step_var, index, end_index, similarity in best_matches:
    #     if index is not None:
    #         print(f"Best match for step '{step_var}' is {tokenizer.decode(tensor[i][index : end_index])} found at index {index} with similarity {similarity:.2f}")
    #     else:
    #         print(f"Failed to find a close match for step '{step_var}'")

    #breakpoint()

    question_buckets = []
    for step_var, index, end_index, similarity in best_matches:
        #breakpoint()
        if index is not None:  
            bucket = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"{BUCKET_ASSIGNING_PROMPT} {stripped_outputs[i]} \nStep: {step_var}"}
                        ],
                        temperature = 0
                    )
            bucket_data = bucket.choices[0].message.content.strip()
            question_buckets.append((step_var, bucket_data)) # tuple of (step, classified bucket)
            if not merged_buckets:
                if bucket_data not in bucket_step_mapping:
                    bucket_step_mapping[bucket_data] = [step_var]
                else:
                    bucket_step_mapping[bucket_data].append(step_var)

    if not merged_buckets:
        merged_buckets = question_buckets
    else:
        for qb in range(len(question_buckets)): # these are the newly created (step, bucket) pairs that we're trying to merge with the existing merged_buckets. We favor the existing named buckets (rather than replacing their names). 
            for j in bucket_step_mapping:
                #breakpoint()
                #random_step_sample = random.choice(bucket_step_mapping[j])
                num_elements = min(len(bucket_step_mapping[j]), 5)
                random_step_sample = random.sample(bucket_step_mapping[j], num_elements)  # these are random steps from the bucket to ask gpt-4o to compare with a new step
                
                yes_no = openai_client.chat.completions.create( 
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{MERGE_BUCKETS_WITH_STEPS_PROMPT} \nExamples from existing bucket: '{random_step_sample}' \n Current step to compare to bucket: '{question_buckets[qb][0]}'"}
                    ],
                    temperature = 0
                )

                # print(f"First step: {random_step_sample}", f"Second step: {question_buckets[i][0]}", yes_no.choices[0].message.content.strip())

                if yes_no.choices[0].message.content.strip() == "Yes":
                    #print(f"First step: {random_step_sample}", f"Second step: {question_buckets[i][0]}", yes_no.choices[0].message.content.strip())
                    question_buckets[qb] = (question_buckets[qb][0], j)
                    break # assuming no duplicate keys in bucket_step_mapping
        
            if question_buckets[qb][1] not in bucket_step_mapping:
                bucket_step_mapping[question_buckets[qb][1]] = [question_buckets[qb][0]] # add the step to the bucket_step_mapping if it's a new bucket
            else:
                bucket_step_mapping[question_buckets[qb][1]].append(question_buckets[qb][0])
        
    answer_bucket_mapping[i] = (stripped_outputs[i], question_buckets)
        # # This code below compares the natural language labels of each bucket with each other and asks if they should be merged.
        # for i in range(len(question_buckets)):
        #     for j in range(len(merged_buckets)):
        #         #breakpoint()
        #         yes_no = openai_client.chat.completions.create(
        #             model="gpt-4o",
        #             messages=[
        #                 {"role": "system", "content": SYSTEM_PROMPT},
        #                 {"role": "user", "content": f"{MERGE_BUCKETS_PROMPT} '{question_buckets[i]}' and '{merged_buckets[j]}'"}
        #             ],
        #             temperature = 0
        #         )
        #         print(f"First bucket: {question_buckets[i]}", f"Second bucket: {merged_buckets[j]}", yes_no.choices[0].message.content.strip())

        #         if yes_no.choices[0].message.content.strip() == "Yes":
        #             question_buckets[i] = merged_buckets[j]
        #             break # assuming no duplicates in merged_buckets
        # merged_buckets = list(set(merged_buckets + question_buckets))
    #buckets.append(question_buckets)

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Save bucket_step_mapping to a file
bucket_step_mapping_file = f"bucket_step_mapping_{timestamp}.txt"
with open(bucket_step_mapping_file, "w") as file:
    for bucket, steps in bucket_step_mapping.items():
        file.write(f"{bucket}: {', '.join(steps)}\n")

# Save answer_bucket_mapping to a file
answer_bucket_mapping_file = f"answer_bucket_mapping_{timestamp}.txt"
with open(answer_bucket_mapping_file, "w") as file:
    for answer, buckets in answer_bucket_mapping.items():
        file.write(f"{answer}: {buckets}\n")


breakpoint()

# "".join(data.split("<STEP SPLIT>")) == stripped_outputs[0]