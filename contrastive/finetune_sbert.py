import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

def load_data(train_file):
    df = pd.read_csv(train_file)
    train_df = df.loc[df.split == 'train']
    val_df = df.loc[df.split == 'val']
    return train_df, val_df

def create_examples(df, limit=None):
    examples = []
    for idx, row in df.iterrows():
        examples.append(InputExample(texts=[row['Reasoning Trace'], row['Positive'][0]], label=1))
        for neg_ex in row['Negative']:
            examples.append(InputExample(texts=[row['Reasoning Trace'], neg_ex], label=0))
        if limit and idx >= limit - 1:
            break
    return examples

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(args.model_name).to(device)

    train_df, val_df = load_data(args.data_file)

    train_examples = create_examples(train_df)
    # val_examples = create_examples(val_df, 3)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.ContrastiveLoss(model=model)

    # val_loader = DataLoader(val_examples, batch_size=64)
    # val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    #     val_examples, batch_size=args.batch_size, name='all-MiniLM-L6-v2-val_trial0', show_progress_bar=True
    # )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        # evaluator=val_evaluator,
        # evaluation_steps=args.evaluation_steps
    )

    model.save('sbert_models/sbert_' + args.model_name + 'b' + str(args.batch_size) + 'e' + str(args.epochs))

def main():
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="S-BERT model name.")
    parser.add_argument("--data_file", type=str, default='data/gsm8k_cl_trans_para1_112_autosplit.csv', help="Path to the CSV data file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps.")
    parser.add_argument("--evaluation_steps", type=int, default=10000, help="Steps between evaluations.")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
