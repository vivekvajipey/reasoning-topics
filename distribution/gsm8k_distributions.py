import argparse
import csv
import numpy as np
import os
import pandas as pd
from pprint import pprint
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from util import get_prompt_message, extract_last_integer, extract_last_number
from util import remove_last_sentence

def generate_responses(model, tokenizer, question, num_samples=2, num_fewshot=3, temp=0.0, direct_prompt=False):
    response_outputs = []
    unique_answers = {}

    messages = get_prompt_message(question, num_fewshot, direct=direct_prompt)
    # messages = [{"role": "user", "content": question } ] # quick for testing
    for i in range(num_samples):
        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(
            input_tensor.to(model.device), 
            max_new_tokens=1000, 
            return_dict_in_generate=True, 
            output_scores=True,
            do_sample=True,
            temperature=temp,
            top_k=40,
        )
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # print(outputs.sequences)

        input_length = input_tensor.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        last_segment = "".join([tokenizer.decode(tok.item()) for tok in generated_tokens[0]]) # bugs when constraining to [0, -10:]
        last_number = extract_last_number(last_segment)
        assert last_number is not None, "No numbers in last segment..."
        last_integer = int(last_number)
        print(f"Extracted {last_integer} from: ", last_segment[-100:])

        unique_answers[last_integer] = 1 + unique_answers.get(last_integer, 0)

        # for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #     # | token | token string | log probability | probability
        #     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy()} | {np.exp(score.cpu().numpy()):.2%}")
        
        response_outputs.append((outputs.sequences, generated_tokens[0], transition_scores[0]))
    
    return response_outputs, unique_answers

def to_tokens_and_logprobs(model, tokenizer, input_ids):
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # Adjust indices to ignore the first token's log prob as it corresponds to the second token
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def calculate_logp_ai_given_q_with_logprobs(model, tokenizer, generated_outputs, unique_answers):
    logp_ai_given_q = {}

    for ai_numeric in unique_answers.keys():
        total_log_prob = 0.0
        count = 0

        for total_tokens, generated_tokens, generated_logprobs in generated_outputs:
            ans_declaration_toks = torch.Tensor(tokenizer(" The answer is " + str(ai_numeric))['input_ids'][1:]) # not adding '.' token or [2] EOS token here
            ans_declaration_toks = ans_declaration_toks.to(torch.int32).to(model.device)
            total_tokens = torch.cat((total_tokens[:, :-1], ans_declaration_toks.unsqueeze(0)), dim=1)
            
            log_probs = to_tokens_and_logprobs(model, tokenizer, total_tokens) 

            # print("\nCalculated Log Probabilities from Concatenated Text:")
            # for sequence in log_probs:
            #     for token, log_prob in sequence:
            #         print(f"Token: {token}, Calculated Log Prob: {log_prob}")

            # Sum the log probabilities of the answer tokens
            answer_log_probs = sum(log_prob for token, log_prob in log_probs[0][-(len(str(ai_numeric))):] if token in str(ai_numeric))
            # answer_log_probs = 0.0
            # for token, log_prob in log_probs[0][-(len(str(ai_numeric))):]:
            #     if token in str(ai_numeric):
            #         print(f"logprob of token '{token}': {log_prob}")
            #         answer_log_probs += log_prob
            # print(f"log P(a_i = {ai_numeric} | r_k = '{rk[7:10]}...', q) = ", answer_log_probs) 
            # print("Original answer for trace (a_k): ", ak)
            # print()
            
            total_log_prob += answer_log_probs
            count += 1

        # Calculate the average log probability for ai
        avg_log_prob = total_log_prob / count if count > 0 else float('inf')
        logp_ai_given_q[ai_numeric] = -avg_log_prob  # Note the negation to get -log p(a_i | q)

    return logp_ai_given_q

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
    parser.add_argument("--num_rows", type=int, default=5, help="Number of rows to process from the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate for each question.")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature setting for the generation process.")
    parser.add_argument("--num_fewshot", type=int, default=3, help="Number of few-shot examples to use for generation.")
    parser.add_argument("--filename", type=str, default="logs/mistral-default-exp.csv", help="Filename for logging the results.")

    args = parser.parse_args()

    START_ROW = args.start_row
    NUM_ROWS = args.num_rows
    NUM_SAMPLES = args.num_samples
    TEMPERATURE = args.temp
    
    NUM_FEWSHOT = args.num_fewshot
    FILENAME = args.filename
    
    DIRECT_PROMPT = True

    gpt35_df = pd.read_csv('../conditional/data/112_gsm8k_gpt35_cot_onesent_responses.csv')

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    total_start = time.time()
    for idx, row in gpt35_df[START_ROW:START_ROW + NUM_ROWS].iterrows():
        start_time = time.time()
        question = row['Question']
        # print("CURRENT QUESTION: ", question)
        generated_outputs, unique_answers = generate_responses(model, tokenizer, question, num_samples=NUM_SAMPLES, num_fewshot=NUM_FEWSHOT, temp=TEMPERATURE, direct_prompt=DIRECT_PROMPT)
        print("UNIQUE ANSWER COUNTS: ", unique_answers)
        logp_ai_given_q = calculate_logp_ai_given_q_with_logprobs(model, tokenizer, generated_outputs, unique_answers)

        log_results_to_csv(row['Problem Number'], unique_answers, logp_ai_given_q, filename=FILENAME)

        for ai, logp in logp_ai_given_q.items():
            print(f"-log p({ai} | q): {logp}")
        print("iteration time (s): ", time.time() - start_time)
    print("total time (min): ",  (time.time() - total_start) / 60)

if __name__ == "__main__":
    main()