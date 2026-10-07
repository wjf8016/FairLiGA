import numpy as np
import torch

from utils import metric, metrics
from params import args

#test_loader: PyTorch 的 DataLoader，每个 batch 返回 (user_ids, pos_item_ids)，通常用于“留一法”测试（每个用户一个正样本）。
def test(test_loader, tar_u2i, user_emb, item_emb):
    topk_list = [10, 20, 30]
    max_k = max(topk_list)

    res = {f'recall@{k}': 0.0 for k in topk_list}
    res.update({f'ndcg@{k}': 0.0 for k in topk_list})

    total_samples = 0

    device = user_emb.device

    for user, item in test_loader:
        batch_size = user.shape[0]
        total_samples += batch_size

        sampled_user, sampled_item = sample_uninter_items(tar_u2i, user, item)

        sampled_user = torch.from_numpy(sampled_user).long().to(device)
        sampled_item = torch.from_numpy(sampled_item).long().to(device)

        user_vectors = user_emb[sampled_user]
        item_vectors = item_emb[sampled_item]
        scores = torch.sum(user_vectors * item_vectors, dim=1)
        scores = scores.view(batch_size, 100)

        sampled_item = sampled_item.view(batch_size, 100)

        for i in range(batch_size):
            ground_truth = [item[i].item()]
            _, topk_idx_all = torch.topk(scores[i], max_k)
            full_ranked = sampled_item[i][topk_idx_all].cpu().tolist()

            for k in topk_list:
                ranked_list = full_ranked[:k]
                recall = metrics.RECALL(ranked_list, ground_truth)
                ndcg = metrics.NDCG(ranked_list, ground_truth)
                res[f'recall@{k}'] += recall
                res[f'ndcg@{k}'] += ndcg

    for key in res:
        res[key] /= total_samples

    for k in topk_list:
        print(f"recall@{k} = {res[f'recall@{k}']:.6f}")
        print(f"ndcg@{k} = {res[f'ndcg@{k}']:.6f}")
    return res

def sample_uninter_items(tar_u2i, batch_users, batch_item):
    sampled_user = np.repeat(batch_users.numpy(), 100)
    sampled_item = np.array([])
    tar_u2i = tar_u2i[batch_users].toarray()
    for i in range(len(batch_users)):
        negset = np.flatnonzero(tar_u2i[i] == 0)
        test_items = np.random.permutation(negset)[:99]
        test_items = np.append(test_items, batch_item[i])
        sampled_item = np.concatenate((sampled_item, test_items))
    return sampled_user, sampled_item
