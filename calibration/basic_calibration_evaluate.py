import os
os.environ['HF_HOME'] = '/scr/vvajipey/.cache/huggingface'
os.environ['HF_HUB'] = '/scr/vvajipey/.cache/huggingface'
from huggingface_hub import login
login("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")

import argparse
import csv
import ast
import numpy as np
import os
import pandas as pd
import torch
from collections import Counter, defaultdict
from difflib import get_close_matches
from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint
import random
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.nn.functional as F

def sum_answer_logprobs(model, tokenizer, input_ids, answer_mask, batch_size=5, run_problem_name="nameless_logprobs", print_logging=False):
    input_ids = input_ids.to(model.device)
    answer_mask = answer_mask.to(model.device)

    start_time = time.time()
    total_summed_logprobs = torch.tensor([]).to(model.device)
    norm_total_summed_logprobs = torch.tensor([]).to(model.device)

    if print_logging:
        print("input_ids (rk_ai) shape: ", input_ids.shape)
        print("input_ids (rk_ai): ", input_ids)

    for i in range(0, input_ids.shape[0], batch_size):
        start_row_idx = i
        end_row_idx = min(i + batch_size, input_ids.shape[0])
        batch_ids = input_ids[start_row_idx:end_row_idx]

        with torch.no_grad():
            outputs = model(batch_ids)
        forward_pass_duration = time.time() - start_time
        if print_logging:
            print("Forward pass duration: ", forward_pass_duration)

        logprobs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # Adjust indices to ignore the first token's log prob as it corresponds to the second token
        logprobs = logprobs[:, :-1, :]
        batch_ids  = batch_ids [:, 1:]

        # get logprobs corresponding to specific input_id tokens (out of all vocab logprob distribution)
        gen_logprobs = torch.gather(logprobs, 2, batch_ids[:, :, None]).squeeze(-1)

        SAVE_LOGPROBS = False
        if SAVE_LOGPROBS:
            torch.save(gen_logprobs, f"logprob_tensors/{run_problem_name}-logprobs.pt")

        masked_logprobs = gen_logprobs * answer_mask[start_row_idx:end_row_idx, 1:].float() # extract logprobs from answer tokens
        if print_logging:
            print("Answer Mask shape: ", answer_mask.shape)            
            print("Answer Mask: ", answer_mask)

        if print_logging:
            print("Masked logprobs shape: ", masked_logprobs.shape)
        nonzero_elements_count = (masked_logprobs != 0).sum(dim=1)
        if print_logging:
            print(masked_logprobs)
            print("Nonzero elements count per row (a_i length): ", nonzero_elements_count)

        batch_summed_logprobs = masked_logprobs.sum(dim=1)
        if print_logging:
            print("Batch summed logprobs: ", batch_summed_logprobs.shape, "\n", batch_summed_logprobs)

        total_summed_logprobs = torch.cat((total_summed_logprobs, batch_summed_logprobs), dim=0)
        norm_total_summed_logprobs = torch.cat((norm_total_summed_logprobs, batch_summed_logprobs / nonzero_elements_count), dim=0)
        
        if print_logging:
            print("summed_probs: ", batch_summed_logprobs.shape)

            for input_sentence, input_probs in zip(batch_ids , masked_logprobs):
            # for input_sentence, input_probs in zip(batch_ids , gen_logprobs): # check all logprobs
                for token, p in zip(input_sentence, input_probs):
                    if token not in tokenizer.all_special_ids:
                        print(f"{tokenizer.decode(token)} ({token}): {p.item()}")

    return total_summed_logprobs, norm_total_summed_logprobs



def main():
    parser = argparse.ArgumentParser(description="Basic Calibration Generation Script")
    parser.add_argument("--model_name", type=str, default="mistral-7b-v0.1", help="Name of the model to use for generation.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples to use for generation.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature setting for the generation process.")
    parser.add_argument("--top_k", type=int, default=40, help="Top k parameter to use for generation.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose printing")
    parser.add_argument("--dataset", type=str, default="commonsense_qa", help="Name of MC dataset to use")
    parser.add_argument("--data_split", type=str, default="train", help="Train/Val/Test")
    parser.add_argument("--start_q", type=int, default=0, help="question index of df")
    parser.add_argument("--num_questions", type=int, default=1, help="number of questions to run")
    args = parser.parse_args()

    run_name = f"{args.model_name}-samples{args.num_samples}-fewshot{args.num_fewshot}-temp{args.temp}-topk{args.top_k}"
    # Load the model and tokenizer
    name2base = {"mistral-7b-v0.1":"mistralai/Mistral-7B-v0.1"}

    base_model_name = name2base[args.model_name]
    print("Loading ", base_model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.generation_config = GenerationConfig.from_pretrained(base_model_name)
    base_model.generation_config.pad_token_id =base_model.generation_config.eos_token_id

    df = pd.read_csv(f"data/{args.dataset}_{args.data_split}.csv")
    df['choices'] = df['choices'].apply(ast.literal_eval)

    for q_num in range(args.start_q, args.start_q + args.num_questions):
        for answer_letter in ['A', 'B', 'C', 'D', 'E']:
            print(f"evaluating for question {q_num}, answer {answer_letter}")
            full_answer = torch.load(f"tensors/full_answer_tensors/csqa_q{q_num}_a{answer_letter}_n{args.num_samples}_t{args.temp}_tk{args.top_k}_logprobs.pt")
            answer_mask = torch.load(f"tensors/answer_masks/csqa_q{q_num}_a{answer_letter}_n{args.num_samples}_t{args.temp}_tk{args.top_k}_answer_mask.pt")
            # instruct_log_probs = torch.load(f"tensors/instruct_log_probs/columbus-csqa_q{q_num}_a{answer_letter}_n{args.num_samples}_instruct_logprobs.pt")

            total_summed_logprobs, norm_total_summed_logprobs = sum_answer_logprobs(base_model, base_tokenizer, full_answer, answer_mask, batch_size=args.num_samples, print_logging=False)

            csv_file_path = f"logs/base_logprobs_{args.dataset}_{args.data_split}_q{q_num}_t{args.temp}_tk{args.top_k}.csv"
            file_exists = os.path.isfile(csv_file_path)

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                if not file_exists:
                    writer.writerow(["question_number", "answer_letter", "total_summed_logprobs", "norm_total_summed_logprobs"])
                for total_logprob, norm_logprob in zip(total_summed_logprobs.tolist(), norm_total_summed_logprobs.tolist()):
                    writer.writerow([q_num, answer_letter, total_logprob, norm_logprob])            

    print("Completed.")

if __name__ == "__main__":
    main()
