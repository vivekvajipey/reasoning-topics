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

FILE_PATH = "/Users/adityatadimeti/reasoning-topics/conditional/data/mistral-7b-v0.1-samples10-fewshot0-temp0.7-topk40-CoT-gsm8k_p9-1.pt"
#FILE_PATH = "/Users/adityatadimeti/reasoning-topics/conditional/data/tensors/mistral-7b-v0.1-samples3-fewshot0-temp0.7-topk40-CoT-gsm8k_p0.pt"
SYSTEM_PROMPT = "You are a skilled mathematical reasoner."
QUESTION = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"

INITIAL_QUESTION_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Categorize the solution into steps using descriptive, specific, and concise natural language labels.   
You should return the steps, in order, separated by commas in one single line. The goal is to call output.split(", ") on the output you generate. 

Question: {QUESTION}
Answer:"""

TEMP_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Your task is to identify where in the solution distinct steps occur, and to separate the solution at the location of each step with <STEP SPLIT>.
You should return the steps, in order, separated by <STEP SPLIT> in one single line. The goal is to call output.split("<STEP SPLIT>") on the output you generate. 
If I concatenate the steps you provide, they should be identical to the original solution.
Do not make the steps highly granular. 
You should NOT distinguish the planning of a step and its execution: for instance, model-outputs that say "First we must do this." and follow it with the actual calculation should remain in the same step.

Here's an example:

Question: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?
Model-generated output: "First, let's find out how much Tina makes for a regular 8-hour shift: $18.00/hour * 8 hours = $144.00

Next, let's calculate Tina's overtime pay. We know that she works 10 hours every day for 5 days, so she works a total of 50 hours. Since she works more than 8 hours per shift, she is eligible for overtime, which is calculated as follows:

Overtime hours = Total hours worked - Regular hours worked
Overtime hours = 50 hours - 40 hours (8 hours per day for 5 days)
Overtime hours = 10 hours

Now, let's calculate Tina's overtime pay:

Overtime pay = Overtime hours * (Hourly wage + Overtime wage) / 2
Overtime pay = 10 hours * ($18.00/hour + $9.00/hour) / 2
Overtime pay = 10 hours * ($27.00/hour)
Overtime pay = $270.00

Finally, let's calculate Tina's total earnings for the 5-day workweek:

Total earnings = Regular earnings + Overtime earnings
Total earnings = $144.00 + $270.00
Total earnings = $614.00

So, Tina makes a total of $614.00 for the 5-day workweek."
<STEP SPLIT> verion: "First, let's find out how much Tina makes for a regular 8-hour shift: $18.00/hour * 8 hours = $144.00 <STEP SPLIT> Next, let's calculate Tina's overtime pay. We know that she works 10 hours every day for 5 days, so she works a total of 50 hours. Since she works more than 8 hours per shift, she is eligible for overtime, which is calculated as follows:

Overtime hours = Total hours worked - Regular hours worked
Overtime hours = 50 hours - 40 hours (8 hours per day for 5 days)
Overtime hours = 10 hours <STEP SPLIT> Now, let's calculate Tina's overtime pay:

Overtime pay = Overtime hours * (Hourly wage + Overtime wage) / 2
Overtime pay = 10 hours * ($18.00/hour + $9.00/hour) / 2
Overtime pay = 10 hours * ($27.00/hour)
Overtime pay = $270.00 <STEP SPLIT> Finally, let's calculate Tina's total earnings for the 5-day workweek:

Total earnings = Regular earnings + Overtime earnings
Total earnings = $144.00 + $270.00
Total earnings = $614.00. So, Tina makes a total of $614.00 for the 5-day workweek."

Notice how intermediate arithmetic calculations are not split into separate steps. Also notice how the step splits leverage the natural language structure of the solution to insert the steps, such as transition words that may be present. Notice how the last step is not ended with a <STEP SPLIT> token. Notice how a separate step is not assigned for stating the final answer.
As a final note: even if one step has intermediate steps that aren't present in the other, but the overarching step is the same, they should be considered the same bucket.

Question: {QUESTION}
Answer:
"""

BUCKET_ASSIGNING_PROMPT = f"""I will give you a math problem, a model-generated solution to the problem, and a substep from the model-generated solution. You should return a descriptive, specific, and concise
natural language label for the substep. The label should NOT comment on the correctness of the substep.

Question: {QUESTION}
Answer:"""

MERGE_BUCKETS_PROMPT = f"""Based on this question: {QUESTION}, would you say these 2 model-generated names of solutions steps refer to the same thing? Return just a "Yes" or "No". The goal is to merge redundant buckets. 
"""

MERGE_BUCKETS_WITH_STEPS_PROMPT = f"""Based on this question: {QUESTION}, does this new model-generated substep belong in this existing list of substeps that have been categorized under the same bucket? Return just a 'Yes' or 'No'  The goal is to merge redundant buckets. Ignore whether the step is correct or not, and focus on the steps the solution is trying to take.
Here's some examples:

Examples from existing bucket: 'Finally, let\'s find Tina\'s total earnings for the 5 days:\nTotal Earnings = Daily Earnings * Number of days\nTotal Earnings = $1080 * 5 = $5400\n\nSo, Tina makes $5400 over 5 days if she works 10 hours every day.'. Name of existing bucket: 'Calculating Total Earnings for the Week'
Current step to compare to bucket: ' Finally, let's calculate Tina's total earnings for the week:\n\nTotal Earnings = Standard Hourly Wage * Standard Hours + Total Overtime Pay\nTotal Earnings = $18.00/hour * 40 hours + $1,350.00\nTotal Earnings = $720.00 + $1,350.00\nTotal Earnings = $2,070.00 '. Name of current step: Calculating Total Weekly Earnings
Desired output: "Yes"

Using the natural language labels as context clues, along with the fact the actual steps themselves try to accomplish the same thing, we see that both steps are trying to calcluate the total earnings for the week using intermediate values. 
Even though the answers have different numbers, the overarching steps are the same, so we categorize it as "Yes".

Below is what you should classify. Return just a 'Yes' or 'No' — remember the goal is to merge redundant buckets. Ignore whether the step is correct or not, and focus on the steps the solution is trying to take.

"""

import re

def extract_text_between_inst(text):
    # Use regex to find all the text between [/INST] and [INST]
    pattern = re.compile(r'\[\/INST\](.*?)\[INST\]', re.DOTALL)
    matches = pattern.findall(text)
    return matches

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

tensor = torch.load(FILE_PATH, map_location="cpu") # Load the file into a tensor
decoded_output = tokenizer.batch_decode(tensor)
response = " ".join(decoded_output)

#parsed_outputs = response.split(["[/INST]"])[1:]
parsed_outputs = extract_text_between_inst(response)
parsed_outputs.append(response.split("[/INST]")[-1]) # Add the last part of the response that doesn't have an [INST] tag after it

stripped_outputs = [output.replace("<s>", "").replace("</s>", "").strip() for output in parsed_outputs] # Remove <s> tokens from the list of outputs

buckets = []
merged_buckets = []

bucket_step_mapping = {} # maps a defined bucket to all the steps that have been assigned to it.
answer_bucket_mapping = {} # maps a defined answer to a list of buckets that have been identified, post per-answer merging. 
random.seed(0)
for i in range(tensor.shape[0]):
    print(i, len(bucket_step_mapping))
    completion = openai_client.chat.completions.create(
            model="gpt-4",
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
                        model="gpt-4",
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
        start = 0
        keys = list(bucket_step_mapping.keys())
        for qb in range(len(question_buckets)): # these are the newly created (step, bucket) pairs that we're trying to merge with the existing merged_buckets. We favor the existing named buckets (rather than replacing their names). 
            #for j in bucket_step_mapping:
            while start < len(keys):
                j = keys[start]
                #breakpoint()
                #random_step_sample = random.choice(bucket_step_mapping[j])
                num_elements = min(len(bucket_step_mapping[j]), 5)
                random_step_sample = random.sample(bucket_step_mapping[j], num_elements)  # these are random steps from the bucket to ask gpt-4o to compare with a new step
                yes_no = openai_client.chat.completions.create( 
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{MERGE_BUCKETS_WITH_STEPS_PROMPT} \nExamples from existing bucket: '{random_step_sample}'. Name of existing bucket: {j} \n Current step to compare to bucket: '{question_buckets[qb][0]}'. Name of current step: {question_buckets[qb][1]}"}
                    ],
                    temperature = 0
                )

                # print(f"First step: {random_step_sample}", f"Second step: {question_buckets[i][0]}", yes_no.choices[0].message.content.strip())

                if yes_no.choices[0].message.content.strip() == "Yes":
                    #print(f"First step: {random_step_sample}", f"Second step: {question_buckets[i][0]}", yes_no.choices[0].message.content.strip())
                    question_buckets[qb] = (question_buckets[qb][0], j)
                    # keys.remove(j) #Assumes one existing bucket cannot categorize 2+ steps in a new solution, so we remove for efficiency.
                    # NO LONGER operating under above assumption. We keep this to account for fact that model can still split into redundant substeps.
                    break # assuming no duplicate keys in bucket_step_mapping 
                else:
                    start += 1
        
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

