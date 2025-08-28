import argparse
import os.path as osp
import os
import os
import time
from src.utils import seed_everything

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['TORCH_SCATTER_CUDA_DETERMINISTIC'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = 'sm_86'

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import sys
from torch_geometric_signed_directed.utils.general.link_split import link_class_split
from torch_geometric_signed_directed.data.signed import load_signed_real_data
from src.model import Piglet
from src.train import train, evaluate
from src.utils import sample_unlabeled_edges, create_spectral_features, reset_edge_classifier_parameters
from collections import defaultdict
import random
from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
import warnings
warnings.filterwarnings("ignore", message="scatter_reduce_cuda does not have a deterministic implementation*")

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mode', type=str, default='pos')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--label_ratio', type=float, default=0.1)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--lamb', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--runs', type=int, default=5)
    return parser.parse_args()
        
args = parameter_parser()
seed_everything(args.seed)


dataset_name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)),
                '..', 'tmp_data')
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
data = load_signed_real_data(
    dataset=dataset_name, root=path).to(device)
data.to_unweighted()
link_data = link_class_split(data, prob_val=0.1, prob_test= 0.1,
                             splits=args.runs, seed=args.seed,
                             task='sign', maintain_connect=True, device=device)

mode = args.mode
model_name = 'piglet'
res = []

for index in list(link_data.keys()):
    
    splited_data = link_data[index]
    nodes_num = data.num_nodes
    edge_index = splited_data['train']['edges']
    edge_sign = splited_data['train']['label'] * 2 - 1
    edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(1)], dim=-1)

    labeled_edge, unlabeled_edge = sample_unlabeled_edges(edge_index_s, label_ratio = args.label_ratio, mode=args.mode, seed = args.seed)
    
    init_emb = create_spectral_features(
                pos_edge_index= labeled_edge[labeled_edge[:, 2] > 0][:, :2].t(),
                neg_edge_index= labeled_edge[labeled_edge[:, 2] < 0][:, :2].t(),
                node_num=nodes_num,
                dim=args.in_dim, 
                seed = args.seed
                ).to(device=device)
    

    unlabeled_edge = unlabeled_edge[:,:2].t().to(device)
    unlabeled_prob = torch.full((unlabeled_edge.shape[1],), 0.5, device=device)

    labeled_edge_s = labeled_edge.to(device)
    pos_edge_index = labeled_edge_s[labeled_edge_s[:, 2] > 0][:, :2].t()
    neg_edge_index = labeled_edge_s[labeled_edge_s[:, 2] < 0][:, :2].t()

    splited_data['train']['labeled_edges'] = labeled_edge[:, :2]
    splited_data['train']['labeled_labels'] = (labeled_edge[:, 2] > 0).long()
    
    model = Piglet(nodes_num, args.in_dim, args.out_dim, lamb=args.lamb, device=device, layer_num=args.layer_num, init_emb = init_emb).to(device)
    
    hid = args.out_dim
    model.edge_classifier = nn.Sequential(
        nn.Linear(2 * args.out_dim, hid),
        nn.ReLU(),
        nn.Linear(hid, 1)
    ).to(device)
    reset_edge_classifier_parameters(model)

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
        
    for epoch in range(args.epochs):

        z, loss = train(model, optimizer, pos_edge_index, neg_edge_index, unlabeled_edge, unlabeled_prob, model_name, scale = args.scale, gamma=args.gamma)
        
        eval_info, _ = evaluate(model,  z, splited_data,  eval_flag='val')
        print(f"[ Epoch {epoch:03d} ] \t Loss : {loss:.4f} \tAUC: {eval_info['auc']:.4f} \tMacro-F1: {eval_info['f1_macro']:.4f}")
    
        if unlabeled_edge.numel() > 0 :
            model.eval()
            with torch.no_grad():
                src, dst = unlabeled_edge
                edge_feat = torch.cat([z[src], z[dst]], dim=1)
                logits = model.edge_classifier(edge_feat).squeeze()
                unlabeled_prob = torch.sigmoid(logits)
 
    test_info, _ = evaluate(model, z, splited_data, eval_flag='test')
    print(f'[Test Result] Loss: {loss:.4f}, '
            f'AUC: {test_info["auc"]:.4f}, F1: {test_info["f1"]:.4f}, MacroF1: {test_info["f1_macro"]:.4f}, MicroF1: {test_info["f1_micro"]:.4f}')
    res.append(test_info)



average_metrics = []
for i in res:
    average_metrics.append(
        (i['auc'], i['f1'], i['f1_macro'], i['f1_micro'])
    )

v = np.array(average_metrics)

print(args)
print("---FINAL RESULT---")
print(f"AUC: {v[:, 0].mean():.4f} ({v[:, 0].std():.4f}) ")
print(f"F1: {v[:, 1].mean():.4f} ({v[:, 1].std():.4f}) ")
print(f"MacroF1: {v[:, 2].mean():.4f} ({v[:, 2].std():.4f}) ")
print(f"MicroF1: {v[:, 3].mean():.4f} ({v[:, 3].std():.4f}) ")

