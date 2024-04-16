import numpy as np
import pandas as pd
from pprint import pprint
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from util import get_prompt_message, extract_last_sentence, extract_last_integer, extract_last_number
from util import remove_last_sentence

def generate_responses(model, tokenizer, question, num_samples=2, num_fewshot=1):
    rk_ak_pairs = []
    unique_answers = {}

    prompt_messages = get_prompt_message(question, num_fewshot)
    for i in range(num_samples):
        prompt_input = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(
            prompt_input,
            max_length=10000,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0,
            top_k=40,
            # return_dict_in_generate=True, 
            # output_scores=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(i, generated_text[-50:])
        
        last_sentence = extract_last_sentence(generated_text)
        # last_integer = extract_last_integer(last_sentence)
        last_integer = extract_last_number(last_sentence)
        if last_integer is None:
            # last_integer = extract_last_integer(generated_text)
            last_integer = extract_last_number(generated_text)

        assert last_integer is not None, "No last integer extracted"
        unique_answers[last_integer] = unique_answers.get(last_integer, 0) + 1
        
        rk_ak_pairs.append((generated_text, last_sentence, last_integer))
    
    return rk_ak_pairs, unique_answers

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(model.device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
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

def calculate_logp_ai_given_q_with_logprobs(model, tokenizer, rk_ak_pairs, unique_answers):
    logp_ai_given_q = {}

    for ai_numeric in unique_answers.keys():
        total_log_prob = 0.0
        count = 0

        for rk, ak, _ in rk_ak_pairs:
            rk_no_last = remove_last_sentence(rk)
            ai_statement = f" The answer is {ai_numeric}."
            ai_num_tokens = len(tokenizer.encode(f"{ai_numeric}."))
            input_text = rk_no_last + ai_statement

            token_logprobs = to_tokens_and_logprobs(model, tokenizer, [input_text])[0]  # Assuming single input

            # Sum the log probabilities of the answer tokens
            # answer_log_probs = sum(log_prob for token, log_prob in token_logprobs[-ai_num_tokens:] if token in str(ai_numeric))
            answer_log_probs = 0.0
            for token, log_prob in token_logprobs[-ai_num_tokens:]:
                if token in str(ai_numeric):
                    print(token, "(", len(token),")", log_prob)
                    answer_log_probs += log_prob
            # print(f"log P(a_i = {ai_numeric} | r_k = '{rk[7:10]}...', q) = ", answer_log_probs) 
            # print("Original answer for trace (a_k): ", ak)
            # print()
            
            total_log_prob += answer_log_probs
            count += 1

        # Calculate the average log probability for ai
        avg_log_prob = total_log_prob / count if count > 0 else float('inf')
        logp_ai_given_q[ai_numeric] = -avg_log_prob  # Note the negation to get -log p(a_i | q)

    return logp_ai_given_q

def main():
    gpt35_df = pd.read_csv('../conditional/data/112_gsm8k_gpt35_cot_onesent_responses.csv')

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    total_start = time.time()
    for idx, row in gpt35_df[:5].iterrows():
        start_time = time.time()
        question = row['Question']
        # print("CURRENT QUESTION: ", question)
        rk_ak_pairs, unique_answers = generate_responses(model, tokenizer, question, num_samples=3, num_fewshot=2)
        print("UNIQUE ANSWER COUNTS: ", unique_answers)
        logp_ai_given_q = calculate_logp_ai_given_q_with_logprobs(model, tokenizer, rk_ak_pairs, unique_answers)

        for ai, logp in logp_ai_given_q.items():
            print(f"-log p({ai} | q): {logp}")
        print("iteration time (s): ", time.time() - start_time)
    print("total time (min): ",  (time.time() - total_start) / 60)

if __name__ == "__main__":
    main()