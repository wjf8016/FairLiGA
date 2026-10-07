import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_pinball_loss


#偏移的sigmoid函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softplus(x):
    return torch.log(1 + torch.exp(x))


def alignment_cos(u, i, x, y):
    batch_user_emb = x[u]
    batch_item_emb = y[i]

    emb1_norm = F.normalize(batch_user_emb, p=2, dim=-1)
    emb2_norm = F.normalize(batch_item_emb, p=2, dim=-1)
    cos_sim = (emb1_norm * emb2_norm).sum(dim=-1)
    l2_loss = 2 - 2 * cos_sim

    return l2_loss.mean()


def uniformity_sampled(embeds, t, sample_size):
    N = embeds.size(0)
    if N <= sample_size:
        x = F.normalize(embeds, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(t).exp().mean()

    idx1 = torch.randperm(N)[:sample_size]
    idx2 = torch.randperm(N)[:sample_size]
    x1 = F.normalize(embeds[idx1], dim=-1)
    x2 = F.normalize(embeds[idx2], dim=-1)

    dist_sq = torch.cdist(x1, x2, p=2).pow(2)
    loss = (t * dist_sq).exp().mean()
    return loss


def alignment2_cos(x, y):
    emb1_norm = F.normalize(x, p=2, dim=-1)
    emb2_norm = F.normalize(y, p=2, dim=-1)
    cos_sim = (emb1_norm * emb2_norm).sum(dim=-1)
    l2_loss = 2 - 2 * cos_sim

    return l2_loss.mean()



