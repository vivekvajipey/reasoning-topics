import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, temperature=0.3):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        # print('Z1 dims: ', z1.shape)
        # print('Z2 dims: ', z2.shape)

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        cos_sim = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature


        n = z1.size(0)
        labels = torch.eye(n).to(z1.device)
        
        # Flatten to match the BCE loss requirements
        cos_sim_flat = cos_sim.view(-1)
        labels_flat = labels.view(-1)
        
        # Apply sigmoid to scale cosine similarity to [0, 1]
        cos_sim_flat_sigmoid = torch.sigmoid(cos_sim_flat)
        
        # Binary Cross Entropy Loss
        loss_fct = nn.BCELoss()
        loss = loss_fct(cos_sim_flat_sigmoid, labels_flat)
        
        return loss