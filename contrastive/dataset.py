import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
import torch

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, sent_transformer, split='train'):
        self.df = pd.read_csv('data/' + csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.sent_transformer = sent_transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Encode the reasoning trace as the "anchor"
        anchor_embedding = self.encode_sentences(row['Reasoning Trace'])
        # print('ANCHOR IN DATASET: ', anchor_embedding.shape)
        # Encode positive and negative examples
        positive_embeddings = self.encode_sentences(row['Positive'])
        # print('POSITIVE IN DATASET: ', positive_embeddings.shape)
        negative_embeddings = self.encode_sentences(row['Negative'])
        return anchor_embedding, positive_embeddings, negative_embeddings

    def encode_sentences(self, sentences):
        return self.sent_transformer.encode(sentences, batch_size=32, show_progress_bar=False)