import torch
from transformers import AutoTokenizer
import os
import openai
from openai import OpenAI
from difflib import SequenceMatcher

# Function to calculate similarity
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

QUESTION_START = "[INST] Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\nA: Let's think step by step. [/INST]"
FILE_PATH = "/Users/adityatadimeti/reasoning-topics/conditional/data/mistral-7b-v0.1-samples10-fewshot0-temp0.7-topk40-CoT-gsm8k_p9-1.pt"
SYSTEM_PROMPT = "You are a skilled mathematical reasoner who can identify the steps in an answer to a problem and provide concise, specific, and descriptive natural language labels for each step."
QUESTION = "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"

INITIAL_QUESTION_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Categorize the solution into steps using descriptive, specific, and concise natural language labels.   
You should return the steps, in order, separated by commas in one single line. The goal is to call output.split(", ") on the output you generate. 

Question: {QUESTION}
Answer:
"""

TEMP_PROMPT = f"""I will give you a math problem and a model-generated solution to the problem. Your task is to identify where in the solution distinct steps occur, and to separate the solution at the location of each step with <STEP SPLIT>.
You should return the steps, in order, separated by <STEP SPLIT> in one single line. The goal is to call output.split("<STEP SPLIT>") on the output you generate. 
If I concatenate the steps you provide, they should be identical to the original solution.

Question: {QUESTION}
Answer:
"""


# def get_stripped_outputs(file_path=FILE_PATH, question_start=QUESTION_START):
#     model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     tensor = torch.load(file_path, map_location="cpu") # Load the file into a tensor
#     decoded_output = tokenizer.batch_decode(tensor)
#     response = " ".join(decoded_output)
#     parsed_outputs = response.split(question_start)[1:]
    
#     stripped_outputs = [output.replace("<s>", "").replace("</s>", "").strip() for output in parsed_outputs] # Remove <s> tokens from the list of outputs
#     return stripped_outputs


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

tensor = torch.load(FILE_PATH, map_location="cpu") # Load the file into a tensor
decoded_output = tokenizer.batch_decode(tensor)
response = " ".join(decoded_output)
parsed_outputs = response.split(QUESTION_START)[1:]

stripped_outputs = [output.replace("<s>", "").replace("</s>", "").strip() for output in parsed_outputs] # Remove <s> tokens from the list of outputs


completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{TEMP_PROMPT} {stripped_outputs[0]}"}
            ],
            temperature = 0
        )
data = completion.choices[0].message.content.strip()
list_output = data.split("<STEP SPLIT>")
best_matches = []

# have to assert that joining them is equal to the original output. 
for step_var in list_output:
    token_length = len(tokenizer.encode(step_var)) - 1 # subtract 1 for the <s> token
    
    best_similarity = 0
    best_index = -1

    found = False
    for start_index in range(1, len(tensor[0]) - token_length):
        decoded_segment = tokenizer.decode(tensor[0][start_index : start_index + token_length])
        similarity = calculate_similarity(decoded_segment, step_var)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = start_index
        

    if best_index != -1:
        best_matches.append((step_var, best_index, best_index + token_length, best_similarity))

for step_var, index, end_index, similarity in best_matches:
    if index is not None:
        print(f"Best match for step '{step_var}' is {tokenizer.decode(tensor[0][index : end_index])} found at index {index} with similarity {similarity:.2f}")
    else:
        print(f"Failed to find a close match for step '{step_var}'")
breakpoint()

# "".join(data.split("<STEP SPLIT>")) == stripped_outputs[0]