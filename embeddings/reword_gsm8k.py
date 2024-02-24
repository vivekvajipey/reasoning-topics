import pandas as pd
import numpy as np
from openai import OpenAI
import openai
import warnings
from dotenv import load_dotenv
import os
import re
import ast
import json

warnings.filterwarnings("ignore")
load_dotenv()
openai.api_key = os.environ.get("REASONING_API_KEY")
if not openai.api_key:
    raise ValueError("API Key for OpenAI not set!")
client = OpenAI(api_key=os.environ.get("REASONING_API_KEY"))

csv_filename = 'data/113_gsm8k_gpt35_cot_responses.csv'
df = pd.read_csv(csv_filename)

df['CoT Steps'] = df['CoT Steps'].apply(ast.literal_eval)

def gpt_query_json(prompt):
    message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt}"}

]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages = message,
        response_format={"type": "json_object"},
        max_tokens=1024,
        temperature=0.0,
        n=1,
        stop=None,
    )

    return response.choices[0].message.content

# df['Paraphrased Question'] = None
# df['Paraphrased CoT Steps'] = None

for i, row in df[:1].iterrows():
    question = row['Question']
    steps = row['CoT Steps']

    reword_prompt = f'''
            Given a math word problem and a step-by-step explanation of the solution, reword the question and the steps of the answer to involve a different "cover story" e.g. change any references to objects, names, actions, months, places etc.
            Include the reworded question and steps as a JSON. Each string element in the original_steps array is a step, ensure that the number of steps in reworded_steps equals the number of steps in original_steps. This means that each step in original_steps should correspond to the step in reworded_steps and the order of overall reasoning should be the same. Do not change the numbers in the problem or solution. Reworded the original_question such that it can be solved with the same sequence of mathematical steps as the original. For example, if one of the original_steps converts from feet to inches, the corresponding string element in the reworded_steps array should also convert from feet to inches (not yards to feet, because this would result in a different result).

            {{
            "original_question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "original_steps": ["First, let's find out how many clips Natalia sold in May. We know that she sold half as many clips in May as she did in April. So, if she sold 48 clips in April, she sold 48/2 = 24 clips in May.", 'To find out how many clips Natalia sold altogether in April and May, we add the number of clips she sold in April to the number of clips she sold in May.   48 + 24 = 72', 'Therefore, Natalia sold a total of 72 clips in April and May.']
            }}

            {{
            "reworded_question": "Marco baked cookies for 48 of his neighbors on Monday, and then he baked half as many cookies on Tuesday. How many cookies did Marco bake in total on Monday and Tuesday?",
            "reworded_steps": ["First, we need to determine the number of cookies Marco baked on Tuesday. It is given that he baked half the amount on Tuesday compared to Monday. Given that he baked 48 cookies on Monday, the number of cookies he baked on Tuesday is 48 divided by 2, which equals 24 cookies.", "To calculate the total number of cookies Marco baked on both Monday and Tuesday, we sum the number of cookies from Monday and Tuesday. That's 48 cookies for Monday plus 24 cookies for Tuesday, yielding a total of 48 + 24 = 72 cookies.", "Hence, on Monday and Tuesday, Marco baked a combined total of 72 cookies."]
            }}
            
            {{
            "original_question": "James creates a media empire.  He creates a movie for $2000.  Each DVD cost $6 to make.  He sells it for 2.5 times that much.  He sells 500 movies a day for 5 days a week.  How much profit does he make in 20 weeks?" 
            "original_steps": ["First, let's calculate the cost of creating the movie. James creates a movie for $2000.", "Next, let's calculate the cost of making each DVD. Each DVD costs $6 to make.", "Now, let's calculate the selling price of each DVD. James sells each DVD for 2.5 times the cost of making it. So, the selling price is 2.5 * $6 = $15.", "Next, let's calculate the number of DVDs sold in a week. James sells 500 movies a day for 5 days a week, so he sells 500 * 5 = 2500 DVDs in a week.", "Now, let's calculate the profit made in a week. The profit made in a week is the selling price per DVD minus the cost per DVD, multiplied by the number of DVDs sold in a week. So, the profit made in a week is ($15 - $6) * 2500 = $22500.", "Finally, let's calculate the profit made in 20 weeks. The profit made in 20 weeks is the profit made in a week multiplied by the number of weeks. So, the profit made in 20 weeks is $22500 * 20 = $450000.", 'Therefore, James makes a profit of $450,000 in 20 weeks.']
            }}

            {{
            "reworded_question": "Lily starts a gardening business. She plants a garden for $2000. Each plant pot costs $6 to prepare. She sells them for 2.5 times that amount. She sells 500 plant pots a day for 5 days a week. What is Lily's profit after 20 weeks?"
            "reworded_steps": ['First, we calculate the initial investment for the gardening tools. Lily spends $2000 on gardening tools.', 'Next, we determine the cost to prepare each set of seedlings. The preparation cost for each set is $6.', "We then figure out the charge for Lily's gardening services. She charges 2.5 times the cost of preparing the seedlings, which means she charges 2.5 * $6 = $15 for each garden service.", 'We calculate the total number of gardens serviced in a week. Lily services 500 gardens a day for 5 days, totaling 500 * 5 = 2500 gardens per week.', 'Now, we compute the weekly profit. The profit for each garden service is the charge minus the cost of seedlings, multiplied by the number of gardens serviced. The weekly profit is ($15 - $6) * 2500 = $22500.', 'To find the total profit after 20 weeks, we multiply the weekly profit by the number of weeks. The 20-week profit is $22500 * 20 = $450000.', "Thus, Lily's profit from her gardening service after 20 weeks is $450,000."]
            }}

            {{
            "original_question": {question},
            "original_steps": {steps}
            }}
            '''

    reworded_json = gpt_query_json(reword_prompt)
    reworded_dict = json.loads(reworded_json)
    
    print("PROBLEM NUMBER ", i)
    print("Original Question: ", question)
    print("Reworded Question: ", reworded_dict["reworded_question"])

    for j, step in enumerate(steps):
        if j >= len(reworded_dict['reworded_steps']):
            print("Reworded steps not same length!")
            print(reworded_dict['reworded_steps'])
            break
        
        print("Original Step: ", step)
        print("Reworded Step: ", reworded_dict["reworded_steps"][j])

    if len(steps) != len(reworded_dict['reworded_steps']):
        break

    df.at[i, 'Paraphrased Question'] = reworded_dict["reworded_question"] 
    df.at[i, 'Paraphrased CoT Steps'] = reworded_dict["reworded_steps"]
    print()
    print()

df.to_csv(csv_filename, index=False)