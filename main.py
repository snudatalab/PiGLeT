import argparse
import os
import os.path as osp
from src.utils import seed_everything
import torch
import numpy as np
from torch_geometric_signed_directed.utils.general.link_split import link_class_split
from torch_geometric_signed_directed.data.signed import load_signed_real_data
from src.model import Piglet
from src.train import train, evaluate
from src.utils import sample_unlabeled_edges, create_spectral_features
from tqdm import tqdm

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--label_ratio', type=float, default=0.1)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--lamb', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=1)
    return parser.parse_args()
        
args = parameter_parser()
dataset_name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)),'..', 'tmp_data')
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

data = load_signed_real_data(dataset=dataset_name, root=path).to(device)
data.to_unweighted()
link_data = link_class_split(data, prob_val=0.1, prob_test= 0.1, splits=5, seed=args.seed, task='sign', maintain_connect=True, device=device)

index = list(link_data.keys())[0]
splited_data = link_data[index]

nodes_num = data.num_nodes
edge_index = splited_data['train']['edges']
edge_sign = splited_data['train']['label'] * 2 - 1
edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(1)], dim=-1)

labeled_edge, unlabeled_edge = sample_unlabeled_edges(edge_index_s, seed = args.seed)

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
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)


for epoch in tqdm(range(args.epochs)):
    z, loss = train(model, optimizer, pos_edge_index, neg_edge_index, unlabeled_edge, unlabeled_prob, gamma=args.gamma)
    
    if unlabeled_edge.numel() > 0:
        model.eval()
        with torch.no_grad():
            src, dst = unlabeled_edge
            edge_feat = torch.cat([z[src], z[dst]], dim=1)
            logits = model.edge_classifier(edge_feat).squeeze(-1)
            unlabeled_prob = torch.sigmoid(logits)

test_info, _ = evaluate(model, z, splited_data, eval_flag='test')


print("\n" + "="*50)
print("RESULT")
print("="*50)
print(f"Dataset : {args.dataset}")
print(f"AUC     : {test_info['auc']:.4f}")
print(f"MacroF1 : {test_info['f1_macro']:.4f}")
print("="*50)