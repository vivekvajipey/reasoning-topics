import pandas as pd
import numpy as np
from openai import OpenAI
import openai
import warnings
from dotenv import load_dotenv
import os
import re

warnings.filterwarnings("ignore")
load_dotenv()
openai.api_key = os.environ.get("REASONING_API_KEY")
if not openai.api_key:
    raise ValueError("API Key for OpenAI not set!")
client = OpenAI(api_key=os.environ.get("REASONING_API_KEY"))

csv_filename = 'data/113_gsm8k_gpt35_cot_responses.csv'
df = pd.read_csv(csv_filename)
df['CoT Steps'] = df['CoT Steps'].apply(ast.literal_eval)

def query_gpt(prompt):
    message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"{prompt}"}

]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages = message,
        max_tokens=1024,
        temperature=0.0,
        n=1,
        stop=None,
    )

    return response.choices[0].message.content

df['CoT Sentences'] = None

for i, row in df.iterrows():
    prompt = f'''I have step-by-step responses that explain the reasoning for solutions for grade school math problems.

    I want to separate these step-by-step solutions into Step 1, Step 2, ..., but current NLP methods like sentence splitting and regex don't take into account the actual logical flow, and can separate only based on punctuation instead.

    Can you help me split the following step-by-step reasoning explanations by logical steps, rather than sentences? There is no limit to the number of steps you may split into, but note that the final answer should be its own step at the end.

    Here are some examples for how I want these reasoning explanations to be split, using || between steps and ** before the first and after the last step:
    Original Explanation:
    First, let's find out how many purple flowers Mark has. We know that there are 80% more purple flowers than yellow flowers. So, we can find this by multiplying the number of yellow flowers by 80% (which is equivalent to multiplying by 0.8) and then adding it to the number of yellow flowers.\n\nYellow flowers: 10\nPurple flowers: 10 + (10 * 0.8) = 10 + 8 = 18\n\nNext, let's find out how many green flowers Mark has. We know that there are only 25% as many green flowers as there are yellow and purple flowers. So, we can find this by multiplying the total number of yellow and purple flowers by 25% (which is equivalent to multiplying by 0.25).\n\nTotal yellow and purple flowers: 10 + 18 = 28\nGreen flowers: 28 * 0.25 = 7\n\nFinally, let's find out the total number of flowers in Mark's garden. We can find this by adding the number of yellow, purple, and green flowers together.\n\nTotal flowers: 10 + 18 + 7 = 35\n\nTherefore, Mark has a total of 35 flowers in his garden.

    Well-split Explanation:
    **
    First, let's find out how many purple flowers Mark has. We know that there are 80% more purple flowers than yellow flowers. So, we can find this by multiplying the number of yellow flowers by 80% (which is equivalent to multiplying by 0.8) and then adding it to the number of yellow flowers.\n\nYellow flowers: 10\nPurple flowers: 10 + (10 * 0.8) = 10 + 8 = 18\n\n
    ||
    Next, let's find out how many green flowers Mark has. We know that there are only 25% as many green flowers as there are yellow and purple flowers. So, we can find this by multiplying the total number of yellow and purple flowers by 25% (which is equivalent to multiplying by 0.25).\n\nTotal yellow and purple flowers: 10 + 18 = 28\nGreen flowers: 28 * 0.25 = 7\n\n,
    ||
    Finally, let's find out the total number of flowers in Mark's garden. We can find this by adding the number of yellow, purple, and green flowers together.\n\nTotal flowers: 10 + 18 + 7 = 35\n\n,
    ||
    Therefore, Mark has a total of 35 flowers in his garden.
    **

    Original Explanation:
    First, let's calculate how many clips Natalia sold in May. We know that she sold half as many clips in May as she did in April. So, we can find this by dividing the number of clips she sold in April by 2.\n\n48 clips / 2 = 24 clips\n\nTherefore, Natalia sold 24 clips in May.\n\nNow, let's calculate how many clips she sold altogether in April and May. We can find this by adding the number of clips she sold in April and the number of clips she sold in May.\n\n48 clips + 24 clips = 72 clips\n\nTherefore, Natalia sold a total of 72 clips in April and May.

    Well-Split Explanation:
    **
    First, let's calculate how many clips Natalia sold in May. We know that she sold half as many clips in May as she did in April. So, we can find this by dividing the number of clips she sold in April by 2.\n\n48 clips / 2 = 24 clips\n\nTherefore, Natalia sold 24 clips in May.\n\n
    ||
    Now, let's calculate how many clips she sold altogether in April and May. We can find this by adding the number of clips she sold in April and the number of clips she sold in May.\n\n48 clips + 24 clips = 72 clips\n\n
    ||
    Therefore, Natalia sold a total of 72 clips in April and May.
    **

    Original Trace:
    To calculate how much Weng earned, we need to first determine how many hours she worked. Since she worked for 50 minutes, we can convert this to hours by dividing it by 60 (since there are 60 minutes in an hour).\n\n50 minutes / 60 minutes = 0.83 hours\n\nTherefore, Weng worked for approximately 0.83 hours.\n\nTo calculate how much she earned, we can multiply the number of hours she worked by her hourly rate.\n\n0.83 hours * $12/hour = $9.96\n\nTherefore, Weng earned $9.96 for babysitting yesterday.

    Well-Split Trace:
    **
    To calculate how much Weng earned, we need to first determine how many hours she worked. Since she worked for 50 minutes, we can convert this to hours by dividing it by 60 (since there are 60 minutes in an hour).\n\n50 minutes / 60 minutes = 0.83 hours\n\nTherefore, Weng worked for approximately 0.83 hours.\n\n
    ||
    To calculate how much she earned, we can multiply the number of hours she worked by her hourly rate.\n\n0.83 hours * $12/hour = $9.96\n\n
    ||
    Therefore, Weng earned $9.96 for babysitting yesterday.
    **


    Explanation:
    Notice how the steps are well-balanced in terms of first stating the logical principal, and then carrying out the math (all within one step). After that reasoning sequence is done, we proceed onto the next large step. Also when splitting the given example into reasoning steps, make sure the final sentence/answer is its own step (the last step), use ** to mark the start and end of the reasoning steps, and use || as a separator between adjacent steps.

    Using these examples, can you split this reasoning trace for me?
    {row['CoT Response']}
    '''
    print("ROW ", i)
    print(row['CoT Response'])
    response = query_gpt(prompt)
    print(response)
    df.at[i, 'CoT Sentences'] = response

df.to_csv(csv_filename, index=False)


def validate_split(original_text, split_text):
    def remove_special_characters(text):
        # Replace the special characters with empty strings
        text = text.replace('**', '').replace('||', '')
        return text

    def compare_texts(text1, text2):
        # Remove special characters from both texts
        cleaned_text1 = remove_special_characters(text1)
        cleaned_text2 = remove_special_characters(text2)

        # Compare and return the result
        return cleaned_text1 == cleaned_text2

    are_texts_same = compare_texts(original_text, split_text)
    print(are_texts_same)

def split_cot(text, delimiters=["||", "**"]):
    delimiters = [re.escape(delimiter) for delimiter in delimiters]
    pattern = '|'.join(delimiters)
    split_text = re.split(pattern, text)
    formatted_text = []
    for section in split_text:
        section = section.replace('\n', ' ').strip()
        if section:
            formatted_text.append(section)
    return formatted_text