import pandas as pd
import torch
import ast

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, sent_transformer, split='train'):
        self.df = pd.read_csv('data/' + csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.df['Positive'] = self.df['Positive'].apply(ast.literal_eval)
        self.df['Negative'] = self.df['Negative'].apply(ast.literal_eval)
        
        self.sent_transformer = sent_transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_embedding = self.encode_sentences(row['Reasoning Trace'])
        positive_embeddings = self.encode_sentences(row['Positive'])
        negative_embeddings = self.encode_sentences(row['Negative'])
        return anchor_embedding, positive_embeddings, negative_embeddings

    def encode_sentences(self, sentences):
        return self.sent_transformer.encode(sentences, batch_size=32, show_progress_bar=False)