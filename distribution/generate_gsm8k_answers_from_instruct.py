import os
os.environ['HF_HOME'] = '/scr/vvajipey/.cache/huggingface'
os.environ['HF_HUB'] = '/scr/vvajipey/.cache/huggingface'
from huggingface_hub import login
login("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")

import argparse
import csv
import numpy as np
import os
import pandas as pd
from pprint import pprint
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from util import get_prompt_message

def generate_gsm8k_answer_tensor(model, tokenizer, question, num_samples, num_fewshot, temp, batch_size=20, top_k=40, direct_prompt=False, verbose=False):
    messages = get_prompt_message(question, num_fewshot, direct_prompt)
    question_vector = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_tensor = question_vector.repeat(num_samples, 1)

    gen_start = time.time()
    outputs = model.generate(
                input_tensor.to(model.device), 
                max_new_tokens=750, 
                return_dict_in_generate=True, 
                output_scores=True,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
            )
    # Print decoded sentences:
    if verbose:
        print(tokenizer.batch_decode(outputs.sequences))
        print("Generation Time: ", time.time() - gen_start)

    return outputs.sequences

def find_last_sentence(tensor):
    batch_size, seq_length = tensor.shape
    digit_tokens = [28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774]
    mask = torch.zeros_like(tensor)
    inverse_mask = torch.zeros_like(tensor)

    for i in range(batch_size):
        row_tensor = tensor[i]

        # Find period indices
        period_indices = (row_tensor == 28723).nonzero()

        seq_end = (row_tensor == 2).nonzero()
        seq_end_index = seq_end[0].item()

        # Filter out periods that are part of decimal numbers
        filtered_indices = []
        for idx in period_indices:
            token_index = idx[0]
            # Remove if period is near end of sequence (it is part of last sentence)
            if seq_end_index - token_index < 5:
                continue
            # Check if the period is not at the start or end of the tensor
            elif token_index > 0 and token_index < seq_length - 1:
                # Check the tokens before and after the period
                token_before = row_tensor[token_index - 1]
                token_after = row_tensor[token_index + 1]
                # Check if both tokens are numeric
                if not (token_before in digit_tokens and token_after in digit_tokens):
                    filtered_indices.append(token_index)

        newline_indices = (row_tensor == 13).nonzero()

        # Determine the last sentence start index
        last_period_index = max(filtered_indices) if filtered_indices else -1
        last_newline_index = newline_indices[-1] if newline_indices.size(0) > 0 else -1
        last_sentence_start_index = max(last_period_index, last_newline_index)

        # Set mask for the last sentence
        if last_sentence_start_index != -1:
            mask[i, last_sentence_start_index+1:seq_end_index+1] = 1
            inverse_mask[i, :last_sentence_start_index+1] = 1

    # Apply mask to get only the last sentences
    last_sentence_only_tensor = tensor * mask
    initial_sentences_tensor = tensor * inverse_mask

    # print("Mask:\n", mask)
    # print("Inverse Mask:\n", inverse_mask)
    # print("Last Sentence Only Tensor:\n", last_sentence_only_tensor)
    # print("Initial Sentences Tensor:\n", initial_sentences_tensor)
    
    return initial_sentences_tensor, last_sentence_only_tensor

def append_suffix_to_prefix(prefixes, suffixes, suffix_index):
    # Extract the last sentence from the first row of suffixes
    last_sentence_first_row = suffixes[suffix_index]
    # Filter out zeros (masked tokens)
    last_sentence_first_row = last_sentence_first_row[last_sentence_first_row != 0]

    # Determine the maximum possible length of the new rows
    max_length = prefixes.size(1) + last_sentence_first_row.size(0)

    # Create a new tensor to hold the result with the expanded size
    new_tensor = torch.zeros((prefixes.size(0), max_length), dtype=prefixes.dtype)
    new_mask = torch.zeros((prefixes.size(0), max_length), dtype=torch.uint8)

    for i in range(prefixes.size(0)):
        # Get the initial sentence tokens from prefixes for the current row
        initial_sentence = prefixes[i]
        initial_sentence = initial_sentence[initial_sentence != 0]

        # Concatenate the initial sentence with the last sentence of the first row
        combined_sentence = torch.cat((initial_sentence, last_sentence_first_row))

        new_tensor[i, :combined_sentence.size(0)] = combined_sentence

        # Create mask for the appended last sentence
        start_index_of_last_sentence = initial_sentence.size(0)
        end_index_of_last_sentence = start_index_of_last_sentence + last_sentence_first_row.size(0)
        new_mask[i, start_index_of_last_sentence:end_index_of_last_sentence] = 1

    return new_tensor, new_mask

def log_results_to_csv(idx, unique_answers, logp_ai_given_q, filename="generation_log.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['question_number', 'unique_answers', 'logp_ai_given_q']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write headers if the file is being created
        
        # Write the data row
        writer.writerow({
            'question_number': idx,
            'unique_answers': str(unique_answers),
            'logp_ai_given_q': str(logp_ai_given_q)
        })

def main():
    parser = argparse.ArgumentParser(description="Run the model to generate responses and calculate log probabilities.")
    parser.add_argument("--start_row", type=int, default=0, help="Starting row for processing the dataset.")
    parser.add_argument("--num_rows", type=int, default=1, help="Number of rows to process from the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate for each question.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature setting for the generation process.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples to use for generation.")
    parser.add_argument("--top_k", type=int, default=40, help="top k parameter to use for generation.")
    parser.add_argument("--direct_prompt", action='store_true', help="Indicates if Direct Prompting should be used instead of CoT.")
    parser.add_argument("--model_name", type=str, default="mistral-7b-v0.1", help="Name of model to test on (should have both instruct and base models)")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose printing")
    args = parser.parse_args()

    run_name = f"{args.model_name}-samples{args.num_samples}-fewshot{args.num_fewshot}-temp{args.temp}-topk{args.top_k}"
    if args.direct_prompt:
        run_name += "-direct"
    else:
        run_name += "-CoT"

    # gpt35_df = pd.read_csv('data/112_gsm8k_gpt35_cot_onesent_responses.csv') # train set selected questions
    gpt35_df = pd.read_csv('data/gsm8kTest.csv')

    name2instruct = {"mistral-7b-v0.1":"mistralai/Mistral-7B-Instruct-v0.1"}

    instruct_model_name = name2instruct[args.model_name]
    print("Loading ", instruct_model_name)
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
    instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
    instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    instruct_model.generation_config = GenerationConfig.from_pretrained(instruct_model_name)
    instruct_model.generation_config.pad_token_id = instruct_model.generation_config.eos_token_id
    
    total_start = time.time()
    for idx, row in gpt35_df[args.start_row : args.start_row + args.num_rows].iterrows():
        try:
            start_time = time.time()

            # question = row['Question']
            question = row['question']
            
            sampled_responses = generate_gsm8k_answer_tensor(model=instruct_model,
                                        tokenizer=instruct_tokenizer,
                                        question=question,
                                        num_samples=args.num_samples,
                                        num_fewshot=args.num_fewshot,
                                        temp=args.temp,
                                        top_k=args.top_k,
                                        direct_prompt=args.direct_prompt,
                                        verbose=args.verbose
                                        )
            
            torch.save(sampled_responses, f"tensors/{run_name}-gsm8k_p{row['index']}.pt")
            print("iteration time (s): ", time.time() - start_time)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    print("total time (min): ",  (time.time() - total_start) / 60)

if __name__ == "__main__":
    main()