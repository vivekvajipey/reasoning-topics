import torch
from transformers import AutoTokenizer

QUESTION_START = "[INST] Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\nA: Let's think step by step. [/INST]"
# Specify the file path
file_path = '/Users/adityatadimeti/reasoning-topics/conditional/data/mistral-7b-v0.1-samples10-fewshot0-temp0.7-topk40-CoT-gsm8k_p9-1.pt'
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the file into a tensor
tensor = torch.load(file_path, map_location="cpu")
decoded_output = tokenizer.batch_decode(tensor)
#For token in tensor[0]: tokenizer.decode(token)
response = " ".join(decoded_output)
parsed_outputs = response.split(QUESTION_START)[1:]
# Remove <s> tokens from the list of outputs
stripped_outputs = [output.replace("<s>", "").replace("</s>", "").strip() for output in parsed_outputs]
print(stripped_outputs)
breakpoint()

#[INST] Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\nA: Let's think step by step. [/INST]







#tokenizer to .decode the tensor of tokens, gets text blobs, then pass them into gpt4, identify steps, identify indices in the tensors that correspond to the step boundaries, then can create a mask corresponding to each step