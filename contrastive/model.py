import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import info_nce

class SimpleContrastiveNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=384):
        super(SimpleContrastiveNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.3, loss_type: str = 'BCE'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.loss_type = loss_type

    def forward(self, anchor, positives, negatives):
        # batch_size, num_negatives, _ = negatives.shape
        
        # print('anchor: ', anchor.shape)
        # print('positives: ', positives.shape)
        # print('negatives: ', negatives.shape)

        if self.loss_type == "InfoNCE":
            loss = info_nce(anchor, positives.squeeze(1), negatives, self.temperature, 'mean', 'paired')
        else:
            loss_types = {'BCE': nn.BCEWithLogitsLoss(), 'CE': nn.CrossEntropyLoss()}
            assert self.loss_type in loss_types, "Loss type used in ContrastiveLoss is not supported"
            z2 = torch.cat((positives, negatives), dim=1)
            z1 = anchor.unsqueeze(1).repeat(1, z2.size(1), 1)
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)

            print("z1: ", z1.shape)
            print("z2: ", z2.shape)
            cos_sim = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
            n = z1.size(0)
            labels = torch.eye(n).to(z1.device)
            cos_sim_flat = cos_sim.view(-1)
            labels_flat = labels.view(-1)
            
            # Adjust loss function to BCEWithLogitsLoss
            loss_fct = loss_types[self.loss_type]
            loss = loss_fct(cos_sim_flat, labels_flat)
        return loss