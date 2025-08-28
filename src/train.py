import argparse
import os.path as osp
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
import math
from torcheval.metrics.functional import (
    binary_auroc,
    multiclass_f1_score as f1_score
)

def train(model, optimizer, pos_edge_index, neg_edge_index, unlabeled_edge, unlabeled_prob, model_name = "sgcn", scale = 0.01, gamma=1):
    model.train()
    optimizer.zero_grad()
    device = model.device

    model = model.to(device=device)
    labeled_pos = pos_edge_index
    labeled_neg = neg_edge_index

    unlabeled_edge = unlabeled_edge.to(device)
    unlabeled_prob = unlabeled_prob.squeeze()
    p_prev = unlabeled_prob.squeeze().detach()
    conf   = 2 * torch.abs(0.5 - p_prev)
            
    pos_weight = torch.cat([
        torch.ones(labeled_pos.shape[1], device=device),
        conf * p_prev                           # w_bal = c·p
    ])
    neg_weight = torch.cat([
        torch.ones(labeled_neg.shape[1], device=device),
        conf * (1 - p_prev)                     # w_unbal = c·(1-p)
        ])

    pos_edge_index = torch.cat([labeled_pos, unlabeled_edge], dim=1)
    neg_edge_index = torch.cat([labeled_neg, unlabeled_edge], dim=1)
    z = model(pos_edge_index, neg_edge_index, pos_weight, neg_weight)
    loss = scale * model.loss(z, labeled_pos, labeled_neg)
    src_un = unlabeled_edge[0]
    dst_un = unlabeled_edge[1]
    h_un = torch.cat([z[src_un], z[dst_un]], dim=1)  
    logits_un =  model.edge_classifier(h_un).squeeze() 
    loss += gamma * F.binary_cross_entropy_with_logits(logits_un, p_prev)
    
    src_pos = labeled_pos[0]  # shape [E_pos]
    dst_pos = labeled_pos[1]
    src_neg = labeled_neg[0]
    dst_neg = labeled_neg[1]

    # positive edges
    h_pos = torch.cat([z[src_pos], z[dst_pos]], dim=1)  # [E_pos, 2*d]
    logits_pos = model.edge_classifier(h_pos).squeeze() # [E_pos]
    target_pos = torch.ones_like(logits_pos)

    # negative edges
    h_neg = torch.cat([z[src_neg], z[dst_neg]], dim=1)
    logits_neg = model.edge_classifier(h_neg).squeeze()
    target_neg = torch.zeros_like(logits_neg)

    loss_clf_pos = F.binary_cross_entropy_with_logits(logits_pos, target_pos)
    loss_clf_neg = F.binary_cross_entropy_with_logits(logits_neg, target_neg)
    total_loss = loss + loss_clf_pos + loss_clf_neg
    
    total_loss.backward()
    optimizer.step()

    return z, total_loss.item()


def evaluate(model, z, splited_data, eval_flag='test'):
    model.eval()
    device = next(model.parameters()).device

    # test edges and labels
    edges  = splited_data[eval_flag]['edges'].to(device)  # [E,2]
    labels = splited_data[eval_flag]['label'].long().to(device)  # {0,1}

    with torch.no_grad():
        src = edges[:,0]
        dst = edges[:,1]
        h   = torch.cat([z[src], z[dst]], dim=1)
        logits = model.edge_classifier(h).squeeze()
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()
        auc = binary_auroc(input=probs, target=labels)

        f1_none = f1_score(num_classes=2, average=None, input=preds, target=labels)
        f1_macro = f1_score(num_classes=2, average='macro', input=preds, target=labels)
        f1_micro = f1_score(num_classes=2, average='micro', input=preds, target=labels)

    eval_info = {
        'auc': auc.item(),
        'f1': f1_none[1].item(),
        'f1_macro': f1_macro.item(),
        'f1_micro': f1_micro.item()
    }
 
    return eval_info, preds
