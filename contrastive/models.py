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
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        cos_sim = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        # Diagonal elements are positive examples, off-diagonal are negative
        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        return loss
