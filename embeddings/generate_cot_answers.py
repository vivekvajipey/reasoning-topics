import csv
import numpy as np
import pandas as pd
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
import openai
load_dotenv()
openai.api_key = os.environ.get("REASONING_API_KEY")
if not openai.api_key:
    raise ValueError("API Key for OpenAI not set!")
client = OpenAI(api_key=os.environ.get("REASONING_API_KEY"))

NUM_NAMES = 10
NUM_QUESTIONS = 110

def query_gpt(prompt, temp):
    message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt}"}

]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages = message,
        max_tokens=1024,
        temperature=temp,
        n=1,
        stop=None,
    )

    return response.choices[0].message.content

def append_to_csv(file_path, fieldnames, row_data):
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_data)

def main():
    # df = pd.read_csv('data/1100_gsm8k_name_swap_prompts.csv')
    # csv_file_path = 'data/1100_gsm8k_gpt35_cot_responses.csv'
    
    df = pd.read_csv('data/100_gsm8k_questions_dataset.csv')
    csv_file_path = 'data/113_gsm8k_gpt35_cot_responses.csv'

    # fieldnames = ['Index', 'Changed Prompt', 'CoT for Changed Prompt', 'Changed Solution', 'Gender', 'Country', 'Name', 'Problem Number']
    fieldnames = ['Problem Number', 'Question', 'Answer', 'CoT Response']
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    count = 0

    for idx, row in df.iterrows():
        print("-----------------------------")
        print('COUNT = ', count)
        print("-----------------------------")
        # q = row['Changed Prompt']
        q = row['question']
        prompt = f"Q:{q}\nLet's think step by step.\nA:"
        response = query_gpt(prompt, temp=0.0)
        print(response)
        df.loc[idx, 'CoT for Changed Prompt'] = response
        # append_to_csv(csv_file_path, fieldnames, {'Index': idx, 'Changed Prompt': q, 'CoT for Changed Prompt': response, 'Gender': row['Gender'], 'Country':row['Country'], 'Problem Number':row['Problem Number'], 'True Answer':row['True Answer']})
        append_to_csv(csv_file_path, fieldnames, {'Problem Number': idx, 'Question': q, 'Answer': row['answer'], 'CoT Response':response})
        count += 1


if __name__ == "__main__":
    main()