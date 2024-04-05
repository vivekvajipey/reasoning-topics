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

def get_gpt_response_info(prompt):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{prompt}"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages = message,
        max_tokens=1024,
        temperature=0.0,
        n=1,
        stop=None,
        logprobs=True,
        top_logprobs=5
    )
    logprobs = [entry.logprob for entry in response.choices[0].logprobs.content]
    tokens = [entry.token for entry in response.choices[0].logprobs.content]
    response_text = response.choices[0].message.content
    return response_text, logprobs, tokens, response

df = pd.read_csv('data/112_gsm8k_gpt35_cot_responses_logprobs.csv')

df['One Sentence Response'] = None
df['onesent_logprobs'] = None
df['onesent_tokens'] = None
# df['onesent_api_response'] = None
df['onesent_api_top_logprobs'] = None

for idx, row in df.iterrows():
    print(idx)
    one_sent_prompt = f"{row['Question']} Answer in one sentence."
    try:
        response_text, logprobs, tokens, response = get_gpt_response_info(one_sent_prompt)
    except Exception as e:
        print(f"Error at iteration {idx}: {str(e)}")
        break
    print(response_text)
    df.at[idx, 'One Sentence Response'] = response_text
    df.at[idx, 'onesent_logprobs'] = logprobs
    df.at[idx, 'onesent_tokens'] = tokens
    # df.at[idx, 'onesent_api_response'] = response
    df.at[idx, 'onesent_api_top_logprobs'] = [[(entry.token, entry.logprob) for entry in response.choices[0].logprobs.content[i].top_logprobs] for i in range(len(response.choices[0].logprobs.content))]

df.to_csv('data/new_gpt_responses.csv', index=False)