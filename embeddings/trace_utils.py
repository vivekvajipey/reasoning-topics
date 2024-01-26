import re
from scipy.spatial import distance
import numpy as np
import random

# from openai import OpenAI
# client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo-1106",
#   response_format={ "type": "json_object" },
#   messages=[
#     {"role": "system", "content": "You split step-by-step explanations for math problems into JSON."},
#     {"role": "user", "content": "Who won the world series in 2020?"}
#   ]
# )
# print(response.choices[0].message.content)


def split_cot(text, delimiters=["||", "**"]):
    """
    Splits a given text into a list of strings based on specified delimiters and removes all newline characters.
    Args:
    text (str): The text to be split.
    delimiters (list): The delimiters used for splitting the text.
    Returns:
    list: A list of strings, each representing a section of the text without newline characters.
    """
    # Escaping any regex special characters in delimiters
    delimiters = [re.escape(delimiter) for delimiter in delimiters]
    # Creating a regex pattern to split by any of the delimiters
    pattern = '|'.join(delimiters)
    # Splitting the text by the pattern
    split_text = re.split(pattern, text)
    # Cleaning each split section, removing newlines, and aligning quotes
    formatted_text = []
    for section in split_text:
        section = section.replace('\n', ' ').strip()
        if section:
            # Adding quotes to the beginning and end of the section if they are missing
            # if not section.startswith('"'):
            #     section = '"' + section
            # if not section.endswith('"'):
            #     section = section + '"'
            formatted_text.append(section)
    return formatted_text

def compute_metrics(embeddings):
    # input: embeddings for a single reasoning trace
    # output: dataframe with metrics computed for each reasoning trace
    metrics = {
        'cosine_similarity': [],
        'euclidean_distance': [],
        'manhattan_distance': [],
        'chebyshev_distance': [],
        'random_cosine_similarity': [],
        'random_euclidean_distance': [],
        'random_manhattan_distance': [],
        'random_chebyshev_distance': [],
    }
    
    # Pairwise Sequential (Adjacent embeddings)
    for i in range(len(embeddings) - 1):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]

        # Calculating various distances and similarities
        cosine_sim = 1 - distance.cosine(emb1, emb2)
        euclidean_dist = distance.euclidean(emb1, emb2)
        manhattan_dist = distance.cityblock(emb1, emb2)
        chebyshev_dist = distance.chebyshev(emb1, emb2)
        
        metrics['cosine_similarity'].append(cosine_sim)
        metrics['euclidean_distance'].append(euclidean_dist)
        metrics['manhattan_distance'].append(manhattan_dist)
        metrics['chebyshev_distance'].append(chebyshev_dist)

    # Random Pairs
    for i in range(len(embeddings)):
        random_index = random.choice([j for j in range(len(embeddings)) if j != i])
        emb_random = embeddings[random_index]
        emb_current = embeddings[i]

        # Calculating distances and similarities
        metrics['random_cosine_similarity'].append(1 - distance.cosine(emb_current, emb_random))
        metrics['random_euclidean_distance'].append(distance.euclidean(emb_current, emb_random))
        metrics['random_manhattan_distance'].append(distance.cityblock(emb_current, emb_random))
        metrics['random_chebyshev_distance'].append(distance.chebyshev(emb_current, emb_random))

    return metrics

