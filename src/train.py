import torch
import torch.nn.functional as F

from torcheval.metrics.functional import (
    binary_auroc,
    multiclass_f1_score as f1_score
)
import torch
import torch.nn.functional as F

from torcheval.metrics.functional import (
    binary_auroc,
    multiclass_f1_score as f1_score
)

def train(model, optimizer, pos_edge_index, neg_edge_index, unlabeled_edge, unlabeled_prob, gamma=1):
    model.train()
    optimizer.zero_grad()
    device = model.device

    model = model.to(device=device)
    labeled_pos = pos_edge_index.to(device)
    labeled_neg = neg_edge_index.to(device)
    unlabeled_edge = unlabeled_edge.to(device)
    unlabeled_prob = unlabeled_prob.squeeze(-1).to(device)
    p_prev = unlabeled_prob.detach()
    conf = 2 * torch.abs(0.5 - p_prev)
    
    pos_weight = torch.cat([torch.ones(labeled_pos.shape[1], device=device), conf * p_prev])
    neg_weight = torch.cat([torch.ones(labeled_neg.shape[1], device=device), conf * (1 - p_prev)])

    pos_edge_index = torch.cat([labeled_pos, unlabeled_edge], dim=1)
    neg_edge_index = torch.cat([labeled_neg, unlabeled_edge], dim=1)
    
    z = model(pos_edge_index, neg_edge_index, pos_weight, neg_weight)
    loss = model.loss(z, labeled_pos, labeled_neg)
    
    src_un = unlabeled_edge[0]
    dst_un = unlabeled_edge[1]
    h_un = torch.cat([z[src_un], z[dst_un]], dim=1)  
    logits_un = model.edge_classifier(h_un).squeeze(-1)
    loss += gamma * F.binary_cross_entropy_with_logits(logits_un, p_prev)
    
    src_pos = labeled_pos[0]
    dst_pos = labeled_pos[1]
    src_neg = labeled_neg[0]
    dst_neg = labeled_neg[1]

    h_pos = torch.cat([z[src_pos], z[dst_pos]], dim=1)  
    logits_pos = model.edge_classifier(h_pos).squeeze(-1) # [수정 2]
    target_pos = torch.ones_like(logits_pos)

    h_neg = torch.cat([z[src_neg], z[dst_neg]], dim=1)
    logits_neg = model.edge_classifier(h_neg).squeeze(-1) # [수정 2]
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
    edges  = splited_data[eval_flag]['edges'].to(device)
    labels = splited_data[eval_flag]['label'].long().to(device)

    with torch.no_grad():
        src = edges[:,0]
        dst = edges[:,1]
        h = torch.cat([z[src], z[dst]], dim=1)
        logits = model.edge_classifier(h).squeeze()
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()
        auc = binary_auroc(input=probs, target=labels)
        f1_macro = f1_score(num_classes=2, average='macro', input=preds, target=labels)

    eval_info = {
        'auc': auc.item(),
        'f1_macro': f1_macro.item(),
    }
 
    return eval_info, preds
