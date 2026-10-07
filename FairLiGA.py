import random
from datetime import datetime
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.AU import alignment_cos, uniformity_sampled, alignment2_cos
from utils.graph import Graph
from utils.dataloader import BPRTrainLoader
from models.bias_encoder import SemiGCN
from models.lightgcn import LightGCN
from utils.loss import *
from utils.fair_evaluate import ranking_evaluate
from utils.wandb_logger import WandbLogger
from params import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
Encode sensitive attribute and generate biased representation
'''


def train_semigcn(sens):
    sens_encoder = SemiGCN(n_users, n_items, n_sens, ori_norm_adj).to(device)
    optimizer = torch.optim.Adam(sens_encoder.parameters(), lr=args.lr)

    loss = 0.
    for _ in tqdm(range(args.sim_epochs)):
        _, _, s_u, _ = sens_encoder()

        expoch_loss = F.cross_entropy(s_u.squeeze(), sens.squeeze())

        optimizer.zero_grad()
        expoch_loss.backward()
        optimizer.step()

        loss = expoch_loss.item()
    print(f'SemiGCN {loss = :.6f}')

    with torch.no_grad():
        e_su, e_si, _, _ = sens_encoder()
        e_su = e_su.detach()
        e_si = e_si.detach()
    return e_su, e_si


def train(train_loader, graph, gcn, opt, e_su, e_si, e_pu, e_pi):
    e_du, e_di = gcn()
    aug_norm_adj = graph.generate_aug_norm_adj(e_du, e_di, e_pu, e_pi)

    loss_lst = {'AU': 0., 'emb': 0., 'ub': 0., 'lb': 0.}
    for u, i, j in tqdm(train_loader):
        e_du, e_di = gcn.oriforward(e_pu, e_pi)

        # feature augmentation
        e_au, e_ai = gcn.feature_aug(aug_norm_adj, e_du, e_di, e_su, e_si)

        users_interaction = e_du[u]
        items_pos_interaction = e_di[i]
        items_neg_interaction = e_di[j]

        emb_loss = embloss(users_interaction, items_pos_interaction, items_neg_interaction)
        reconn_loss, _ = calc_bpr_loss(e_au, e_ai, u, i, j)

        loss_alignment = alignment_cos(u, i, e_du, e_di)

        loss_uniformity = (uniformity_sampled(users_interaction, args.t,
                                              sample_size=args.batch_size) + uniformity_sampled(items_pos_interaction,
                                                                                                args.t,
                                                                                                sample_size=args.batch_size)) / 2
        auloss = loss_alignment + args.au_leg * loss_uniformity

        au_loss = auloss + reconn_loss * args.reconn_reg
        emb_loss = emb_loss * args.l2_reg

        lb_aug1 = alignment2_cos(e_du, e_au) + args.au_leg * (
                    uniformity_sampled(e_du, args.t, sample_size=args.batch_size) + uniformity_sampled(e_au, args.t,
                                                                                                       sample_size=args.batch_size)) / 2
        lb_aug2 = alignment2_cos(e_di, e_ai) + args.au_leg * (
                    uniformity_sampled(e_di, args.t, sample_size=args.batch_size) + uniformity_sampled(e_ai, args.t,
                                                                                                       sample_size=args.batch_size)) / 2
        lb_aug3 = alignment2_cos(e_au, e_du) + args.au_leg * (
                    uniformity_sampled(e_du, args.t, sample_size=args.batch_size) + uniformity_sampled(e_au, args.t,
                                                                                                       sample_size=args.batch_size)) / 2
        lb_aug4 = alignment2_cos(e_ai, e_di) + args.au_leg * (
                    uniformity_sampled(e_di, args.t, sample_size=args.batch_size) + uniformity_sampled(e_ai, args.t,
                                                                                                       sample_size=args.batch_size)) / 2
        lb = (lb_aug1 + lb_aug2 + lb_aug3 + lb_aug4) * args.lb_reg

        ub = calc_hsic_loss(e_au[torch.unique(u)], e_su[torch.unique(u)], args.sigma) * args.ub_reg

        loss = au_loss + emb_loss + lb + ub

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_lst['AU'] += au_loss.item()
        loss_lst['emb'] += emb_loss.item()
        loss_lst['ub'] += ub.item()
        loss_lst['lb'] += lb.item()

    loss_lst['AU'] = loss_lst['AU'] / len(train_loader)
    loss_lst['emb'] = loss_lst['emb'] / len(train_loader)
    loss_lst['ub'] = loss_lst['ub'] / len(train_loader)
    loss_lst['lb'] = loss_lst['lb'] / len(train_loader)
    return ', '.join([f'{name}_loss={value:.6f}' for name, value in loss_lst.items()])


if __name__ == '__main__':
    set_random_seed()

    with open(f'./data/{args.dataset}/process/process.pkl', 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)
    n_sens = np.unique(user_side_features['gender']).shape[0]
    u_sens = torch.tensor(user_side_features['gender'], dtype=torch.long).to(device)

    train_dataset = BPRTrainLoader(train_set, train_u2i, n_items)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8)
    graph = Graph(n_users, n_items, train_u2i)
    ori_norm_adj = graph.get_norm_adj(train_u2i)

    gcn = LightGCN(n_users, n_items, ori_norm_adj).to(device)
    opt = torch.optim.Adam(gcn.parameters(), lr=args.lr)

    print(f'{datetime.now()}: Loading pre-trained data encoded by LightGCN...')

    pregcn = LightGCN(n_users, n_items, ori_norm_adj).to(device)
    lightgcn = torch.load(f'./param/pre_tran_ml-1m.pth')
    pregcn.load_state_dict(lightgcn)

    e_pu, e_pi = pregcn.forward()
    e_pu = e_pu.detach().to(device)
    e_pi = e_pi.detach().to(device)
    print('Pre-trained data loaded successfully.')

    # Pretrain to learn e_su and e_si, where e_su is the biased emb
    e_su, e_si = train_semigcn(u_sens)

    if args.wandb:
        wandb_logger = WandbLogger()
    start_time = datetime.now()
    ndcgbest = 0
    for epoch in range(1, args.num_epochs + 1):
        print(f'\n{datetime.now()} Trainning Epoch {epoch}:')
        losses = train(train_loader, graph, gcn, opt, e_su, e_si, e_pu, e_pi)
        print(f'{losses}\nEpoch {epoch} Evaluation Metrics:')
        with torch.no_grad():
            user_emb, item_emb = gcn()
            test_res = ranking_evaluate(
                user_emb.detach().cpu().numpy(),
                item_emb.detach().cpu().numpy(),
                n_users,
                n_items,
                train_u2i,
                test_u2i,
                u_sens.cpu().numpy(),
            )
        end_time = datetime.now()
        if ndcgbest < test_res["ndcg@10"]:
            ndcgbest = test_res["ndcg@10"]
            print(f'epoch {epoch} with ndcg@10: {ndcgbest:.6f} at {end_time - start_time}')
        if args.wandb:
            wandb_logger.log_metrics(epoch, test_res, gcn)
