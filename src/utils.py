import os
import random
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from torch_geometric.utils import coalesce

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_unlabeled_edges(edge_index_s, device='cpu', seed:int=None):
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    train_edges = edge_index_s[:, :2].to(device).long()
    train_labels = edge_index_s[:, 2].to(device)
    positive_edges = train_edges[train_labels > 0]
    negative_edges = train_edges[train_labels < 0]
    num_positive_labeled = int(len(positive_edges) * (0.1))
    num_negative_labeled = int(len(negative_edges) * (0.1))
    pos_labeled_idx = torch.randperm(len(positive_edges), device=device)[:num_positive_labeled]
    neg_labeled_idx = torch.randperm(len(negative_edges), device=device)[:num_negative_labeled]

    labeled_edges = torch.cat([positive_edges[pos_labeled_idx], negative_edges[neg_labeled_idx]], dim=0).long()
    labeled_signs = torch.cat([
        torch.ones(num_positive_labeled, device=device),
        - torch.ones(num_negative_labeled, device=device)
    ], dim=0)
    
    pos_all = set(range(len(positive_edges)))
    neg_all = set(range(len(negative_edges)))
    pos_labeled_set = set(pos_labeled_idx.tolist())
    neg_labeled_set = set(neg_labeled_idx.tolist())
    pos_unlabeled_idx = torch.tensor(sorted(list(pos_all - pos_labeled_set)), device=device)
    neg_unlabeled_idx = torch.tensor(sorted(list(neg_all - neg_labeled_set)), device=device)
    
    unlabeled_edges = torch.cat([
        positive_edges[pos_unlabeled_idx] if len(pos_unlabeled_idx) > 0 else torch.empty((0, 2), device=device),
        negative_edges[neg_unlabeled_idx] if len(neg_unlabeled_idx) > 0 else torch.empty((0, 2), device=device)
    ], dim=0).long()
    
    unlabeled_signs = torch.empty((len(unlabeled_edges),), device=device).long()
    labeled_edge = torch.cat([labeled_edges, labeled_signs.unsqueeze(-1).long()], dim=-1)
    unlabeled_edge = torch.cat([unlabeled_edges, unlabeled_signs.unsqueeze(-1).long()], dim=-1)
    
    return labeled_edge, unlabeled_edge



def create_spectral_features(
    pos_edge_index: torch.LongTensor,
    neg_edge_index: torch.LongTensor,
    node_num: int,
    dim: int,
    seed: int,
) -> torch.FloatTensor:

    edge_index = torch.cat(
        [pos_edge_index, neg_edge_index], dim=1)
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))
    
    pos_val = torch.full(
        (pos_edge_index.size(1), ), 2, dtype=torch.float)
    neg_val = torch.full(
        (neg_edge_index.size(1), ), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N)
    val = val - 1

    # Borrowed from:
    # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))
    # svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd = TruncatedSVD(n_components=dim, n_iter=128, random_state=seed)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)

