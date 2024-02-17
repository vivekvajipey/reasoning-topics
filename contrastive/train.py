from sentence_transformers import SentenceTransformer
from models import SimpleContrastiveNetwork, ContrastiveLoss
import torch
import numpy as np
import pandas as pd
import ast

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('data/simclr_data.csv')
df['Positive'] = df['Positive'].apply(ast.literal_eval)
df['Negative'] = df['Negative'].apply(ast.literal_eval)

# Function to encode sentences to embeddings
def encode_sentences(sentences):
    return model.encode(sentences, batch_size=32)

# Prepare dataset
def prepare_data(df):
    pos_embeddings = []
    neg_embeddings = []
    for _, row in df.iterrows():
        pos_sentences = row['Positive']
        neg_sentences = row['Negative']
        # Encode sentences
        pos_embeddings.append(encode_sentences(pos_sentences))
        neg_embeddings.append(encode_sentences(neg_sentences))
    return np.array(pos_embeddings), np.array(neg_embeddings)

pos_embeddings, neg_embeddings = prepare_data(df)
