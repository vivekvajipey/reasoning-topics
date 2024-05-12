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
from util import set_seed
from util import print_tensors_on_cuda_gpu, print_tensors_on_mps_gpu

TENSOR_PRINT = False

def find_last_sentence(tensor, verbose=False):
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
        inst_indices = (row_tensor == 16289).nonzero()

        # Determine the last sentence start index
        last_period_index = max(filtered_indices) if filtered_indices else -1
        last_newline_index = newline_indices[-1] if newline_indices.size(0) > 0 else -1
        last_inst_index = inst_indices[-1] + 1 if newline_indices.size(0) > 0 else -1 # +1 to include ']' after 'INST' 
        last_sentence_start_index = max(last_period_index, last_newline_index, last_inst_index)

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

def append_suffix_to_prefix(prefixes, suffixes, suffix_index, tokenizer, verbose=False):
    # Extract the last sentence from the first row of suffixes
    
    last_sentence_first_row = suffixes[suffix_index]
    # Filter out zeros (masked tokens)
    last_sentence_first_row = last_sentence_first_row[last_sentence_first_row != 0]
    
    if verbose:
        # start_token_id = tokenizer.bos_token_id
        decoded_suffix = tokenizer.decode(last_sentence_first_row) 
        print("SUFFIXES INDEX", suffix_index)
        print(decoded_suffix.replace("<unk>", "")) 

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
        new_mask[i, start_index_of_last_sentence:end_index_of_last_sentence - 1] = 1 # end_index_of_last_sentence - 1 to exclude EOS (2) token 

    return new_tensor, new_mask


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

        if TENSOR_PRINT:
           print("tensor printing batch ", i)
           print_tensors_on_mps_gpu() 
        
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

        SAVE_LOGPROBS = True
        if SAVE_LOGPROBS:
            torch.save(gen_logprobs, f"tensors/{run_problem_name}-logprobs.pt")

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

            # batch = []
            # for input_sentence, input_probs in zip(batch_ids , masked_logprobs):
            for input_sentence, input_probs in zip(batch_ids , gen_logprobs): # check all logprobs
                # text_sequence = []
                for token, p in zip(input_sentence, input_probs):
                    if token not in tokenizer.all_special_ids:
                        # print((tokenizer.decode(token), p.item()))
                        print(f"{tokenizer.decode(token)} ({token}): {p.item()}")
                        # text_sequence.append((tokenizer.decode(token), p.item()))
                # batch.append(text_sequence)

    print("TOTAL SUMMED LOGPROBS: ", total_summed_logprobs)

    return total_summed_logprobs, norm_total_summed_logprobs


def log_results_to_csv(idx, tokenizer, problem_number, reasoning_steps, answer_statements, all_neg_logp_ai_given_q, entropy, norm_entropy, filename="nameless_generation_log.csv"):
    print("logging to ", filename)
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['question_number', 'reasoning_steps', 'answer_statements', 'all_neg_logp_ai_given_q', 'entropy', 'normalized_entropy']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write headers if the file is being created
        
        writer.writerow({
            'question_number': problem_number,
            'reasoning_steps': [rs.replace("<unk>", "") for rs in tokenizer.batch_decode(reasoning_steps)],
            'answer_statements': [a.replace("<unk>", "") for a in tokenizer.batch_decode(answer_statements)],
            'all_neg_logp_ai_given_q': all_neg_logp_ai_given_q.tolist(),
            'entropy': entropy.item(),
            'normalized_entropy': norm_entropy.item()
        })

def main():
    set_seed(0)

    parser = argparse.ArgumentParser(description="Run the model to generate responses and calculate log probabilities.")
    parser.add_argument("--start_row", type=int, default=0, help="Starting row for processing the dataset.")
    parser.add_argument("--num_rows", type=int, default=1, help="Number of rows to process from the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate for each question.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature setting for the generation process.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples to use for generation.")
    parser.add_argument("--top_k", type=int, default=40, help="top k parameter to use for generation.")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size for forward pass when summing logprobs")
    parser.add_argument("--direct_prompt", action='store_true', help="Indicates if Direct Prompting should be used instead of CoT.")
    parser.add_argument("--model_name", type=str, default="mistral-7b-v0.1", help="Name of model to test on (should have both instruct and base models)")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose printing")
    args = parser.parse_args()

    run_name = f"{args.model_name}-samples{args.num_samples}-fewshot{args.num_fewshot}-temp{args.temp}-topk{args.top_k}"
    if args.direct_prompt:
        run_name += "-direct"
    else:
        run_name += "-CoT"

    gpt35_df = pd.read_csv('data/112_gsm8k_gpt35_cot_onesent_responses.csv')

    # name2instruct = {"mistral-7b-v0.1":"mistralai/Mistral-7B-Instruct-v0.1"}
    name2base = {"mistral-7b-v0.1":"mistralai/Mistral-7B-v0.1"}

    base_model_name = name2base[args.model_name]
    print("Loading ", base_model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.generation_config = GenerationConfig.from_pretrained(base_model_name)
    base_model.generation_config.pad_token_id =base_model.generation_config.eos_token_id
    base_device_map = base_model.hf_device_map

    total_start = time.time()

    if TENSOR_PRINT:
        print("--------------------- BEFORE DF LOOP ---------------------")
        print_tensors_on_mps_gpu()

    for idx, row in gpt35_df[args.start_row : args.start_row + args.num_rows].iterrows():
        try:
            start_time = time.time()

            problem_number = row['Problem Number']
            # print("CURRENT QUESTION: ", question)
            sampled_responses = torch.load(f"tensors/{run_name}-gsm8k_p{row['Problem Number']}.pt")
            
            for module_name, module in base_model.named_modules():
                if module_name in base_device_map:
                    module.to(base_device_map[module_name])

            if args.verbose:
                fls_start = time.time()
            reasoning_steps, answer_statements = find_last_sentence(sampled_responses)
            if args.verbose:
                fls_total_time = time.time() - fls_start
                print(f"find_last_sentence took {fls_total_time} seconds")

            if args.verbose:
                decoded_prefixes = base_tokenizer.batch_decode(reasoning_steps)
            
                print("PREFIXES: ")
                for i, p in enumerate(decoded_prefixes):
                    print(i)
                    print(p.replace("<unk>", ""))
            
            num_ai = answer_statements.size(0)
            print('num a_i: ', num_ai)

            all_surprisals_ai_given_q = torch.tensor([])
            all_norm_surprisals_ai_given_q = torch.tensor([])
            for ai_idx in range(num_ai):
                if args.verbose:
                    astp_start = time.time()
                rk_ai, ai_mask = append_suffix_to_prefix(reasoning_steps, answer_statements, ai_idx, base_tokenizer, args.verbose)
                if args.verbose:
                    astp_total_time = time.time() - astp_start
                    print(f"append_suffix_to_prefix took {astp_total_time} seconds")

                if args.verbose:
                    sal_start = time.time()
                logprobs_ai_given_rk_q, norm_logprobs_ai_given_rk_q = sum_answer_logprobs(
                                                                        base_model, 
                                                                        base_tokenizer, 
                                                                        rk_ai, 
                                                                        ai_mask, 
                                                                        batch_size=args.batch_size, 
                                                                        run_problem_name=f"{run_name}-gsm8k_p{row['Problem Number']}",
                                                                        print_logging=True
                                                                    )
                print("logprobs_ai_given_rk_q: ", logprobs_ai_given_rk_q)
                print("norm_logprobs_ai_given_rk_q: ", norm_logprobs_ai_given_rk_q)
                if args.verbose:
                    sal_total_time = time.time() - sal_start
                    print(f"sum_answer_logprobs took {sal_total_time} seconds")

                # print("logprobs_ai_given_rk_q: ", logprobs_ai_given_rk_q)

                # Calculate -log(P(a_i | q))
                logprob_a_i_given_q = torch.logsumexp(logprobs_ai_given_rk_q, dim=0) - torch.log(torch.tensor(num_ai)) # log ( (1 / num_ai) * sum of (exp ())
                surprisal_a_i_given_q = -logprob_a_i_given_q
                print(f"-log p(a_{ai_idx} | q): {surprisal_a_i_given_q}")
                all_surprisals_ai_given_q = torch.cat((all_surprisals_ai_given_q, torch.tensor([surprisal_a_i_given_q])))

                # Calculate -log(P(a_i | q)) using normalized log probabilities
                norm_logprob_a_i_given_q = torch.logsumexp(norm_logprobs_ai_given_rk_q, dim=0) - torch.log(torch.tensor(num_ai))
                norm_surprisal_a_i_given_q = -norm_logprob_a_i_given_q
                all_norm_surprisals_ai_given_q = torch.cat((all_norm_surprisals_ai_given_q, torch.tensor([norm_surprisal_a_i_given_q])))

            print("iteration time (s): ", time.time() - start_time)

            
            entropy = torch.mean(all_surprisals_ai_given_q)
            print(f"QUESTION {problem_number} ENTROPY: {entropy}")

            norm_entropy = torch.mean(all_norm_surprisals_ai_given_q)
            print(f"Normalized QUESTION {problem_number} ENTROPY: {norm_entropy}")
            log_results_to_csv(idx, base_tokenizer, problem_number, reasoning_steps, answer_statements, all_surprisals_ai_given_q, entropy, norm_entropy,filename=f"logs/{run_name}_logs.csv")
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    print("total time (min): ",  (time.time() - total_start) / 60)

if __name__ == "__main__":
    main()