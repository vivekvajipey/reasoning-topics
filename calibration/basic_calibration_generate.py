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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.nn.functional as F

def find_last_occurrence(tensor, target, offset=2, repeated=False):
    mask = (tensor == target)
    last_occurrence_indices = []
    
    for i in range(tensor.shape[0]):
        indices = torch.nonzero(mask[i], as_tuple=False).squeeze()
        last_occurrence_index = -1
        if indices.numel() > 0:
            last_occurrence_index = indices[-1].item() + offset 
        last_occurrence_indices.append(last_occurrence_index)
        if repeated:
            return last_occurrence_indices * tensor.shape[0]
    
    return last_occurrence_indices

def mask_before_last_occurrence(tensor, target, offset=2):
    last_occurrence_indices = find_last_occurrence(tensor, target, offset, repeated=True)
    
    mask = torch.ones_like(tensor, dtype=torch.bool)
    
    for i in range(tensor.shape[0]):
        if last_occurrence_indices[i] != -1:
            mask[i, :last_occurrence_indices[i]] = 0
    return mask

def generate_answer_justification(model, tokenizer, df, q_num, n_samples, answer_letter, temp=0.7, top_k=40, split='train', dataset='commonsense_qa'):
    df = pd.read_csv(f"data/{dataset}_{split}.csv")
    df['choices'] = df['choices'].apply(ast.literal_eval)

    prompt = f"""Question: {df['question'][q_num]}
        Choices:
        (A) {df['choices'][q_num]['text'][0]}
        (B) {df['choices'][q_num]['text'][1]}
        (C) {df['choices'][q_num]['text'][2]}
        (D) {df['choices'][q_num]['text'][3]}
        (E) {df['choices'][q_num]['text'][4]}
        Answer:"""

    chat = [{"role":"user", "content": prompt}]

    answer_prefix = tokenizer.encode(
        tokenizer.apply_chat_template(chat, tokenize=False) + f" The answer is ({answer_letter}).", 
        return_tensors='pt', 
        add_special_tokens=False
        )
    input_tensor = answer_prefix.repeat(n_samples, 1)

    with torch.no_grad():
        outputs = model.generate(
                    input_tensor.to(model.device),
                    min_new_tokens=0,
                    max_new_tokens=1000,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    temperature=temp,
                    top_k=top_k,
                )

    full_answer = outputs.sequences

    answer_mask = mask_before_last_occurrence(full_answer, 16289, offset=2)
    instruct_log_probs = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    torch.save(full_answer, f"tensors/full_answer_tensors/csqa_q{q_num}_a{answer_letter}_n{n_samples}_t{temp}_tk{top_k}_logprobs.pt")
    torch.save(answer_mask, f"tensors/answer_masks/csqa_q{q_num}_a{answer_letter}_n{n_samples}_t{temp}_tk{top_k}_answer_mask.pt")
    torch.save(instruct_log_probs, f"tensors/instruct_log_probs/csqa_q{q_num}_a{answer_letter}_n{n_samples}_t{temp}_tk{top_k}_instruct_logprobs.pt")

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
    name2instruct = {"mistral-7b-v0.1": "mistralai/Mistral-7B-Instruct-v0.1"}
    instruct_model_name = name2instruct[args.model_name]
    print("Loading ", instruct_model_name)
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
    instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
    instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    instruct_model.generation_config = GenerationConfig.from_pretrained(instruct_model_name)
    instruct_model.generation_config.pad_token_id = instruct_model.generation_config.eos_token_id

    df = pd.read_csv(f"data/{args.dataset}_{args.data_split}.csv")
    df['choices'] = df['choices'].apply(ast.literal_eval)

    for q_num in range(args.start_q, args.start_q + args.num_questions):
        for answer_letter in ['A', 'B', 'C', 'D', 'E']:
            print(f"generating for question {q_num}, answer {answer_letter}")
            generate_answer_justification(
                model=instruct_model, 
                tokenizer=instruct_tokenizer,
                df=df, 
                q_num=q_num, 
                n_samples=args.num_samples, 
                answer_letter=answer_letter, 
                temp=args.temp, 
                top_k=args.top_k
                )

    print("Generation completed and saved.")

if __name__ == "__main__":
    main()
