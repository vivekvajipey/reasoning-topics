import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from dataset import ContrastiveDataset
from model import SimpleContrastiveNetwork, ContrastiveLoss
from sentence_transformers import SentenceTransformer

def train(args, model, train_loader, val_loader, criterion, optimizer, device, run_name):
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            
            anchor_in, positives_in, negatives_in = data
            anchor_in, positives_in, negatives_in = anchor_in.to(device), positives_in.to(device), negatives_in.to(device)

            # print(anchor_in.shape)
            # print(positives_in.shape)
            # print(negatives_in.shape)

            anchor_out = model(anchor_in)
            positives_out = model(positives_in)
            negatives_out = model(negatives_in)

            # print(anchor_out.shape)
            # print(positives_out.shape)
            # print(negatives_out.shape)

            combined_out = torch.cat((positives_out, negatives_out), dim=0)
            # print('combined shape: ', combined_out.shape) 
            # # print('repeated shape: ', anchor_out.repeat(2, 1).shape)

            loss = criterion(anchor_out.repeat(2, 1), combined_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_models/best_{run_name}.pth')

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            anchor_in, positives_in, negatives_in = data
            anchor_in, positives_in, negatives_in = anchor_in.to(device), positives_in.to(device), negatives_in.to(device)
            
            anchor_out = model(anchor_in)
            positives_out = model(positives_in)
            negatives_out = model(negatives_in)

            combined_out = torch.cat((positives_out, negatives_out), dim=0)

            loss = criterion(anchor_out.repeat(2, 1), combined_out)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def generate_run_name(args):
    run_name = "cl_transform"
    run_name += f"-{args.csv_path.split('.')[0]}"
    run_name += f"-b{args.batch_size}"
    run_name += f"-e{args.epochs}"
    run_name += f"-lr{args.lr}"
    run_name += f"-lt{args.loss_temp}"
    return run_name

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # if torch.backends.mps.is_available():
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleContrastiveNetwork().to(device)
    criterion = ContrastiveLoss(temperature=args.loss_temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2') 

    train_dataset = ContrastiveDataset(args.csv_path, sent_transformer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ContrastiveDataset(args.csv_path, sent_transformer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    run_name = generate_run_name(args)

    if args.use_wandb:
        wandb.init(project="gsm8k-cl-transform", name=run_name, config=args)

    train(args, model, train_loader, val_loader, criterion, optimizer, device, run_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Simple Contrastive Network')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--loss_temp', type=float, default=0.3, help='contrastive loss temperature parameter')
    parser.add_argument('--csv_path', type=str, default='gsm8k_cl_trans_para0_47_autosplit.csv', help='dataset csv file path, data folder implied')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='log run to WandB') 

    args = parser.parse_args()
    main(args)