import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from typing import Optional, Union
from torch_geometric.typing import PairTensor, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_geometric_signed_directed.utils.signed import Sign_Structure_Loss

class PigletConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = True,
        add_self_loops: bool = True,  
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.add_self_loops = add_self_loops
        self.norm_emb = norm_emb
        self.lin_b = torch.nn.Linear(in_dim, out_dim, bias)
        self.lin_u = torch.nn.Linear(in_dim, out_dim, bias)
        self.alpha_b2b = torch.nn.Linear(self.out_dim * 2, 1)
        self.alpha_u2b = torch.nn.Linear(self.out_dim * 2, 1)
        self.alpha_u2u = torch.nn.Linear(self.out_dim * 2, 1)
        self.alpha_b2u = torch.nn.Linear(self.out_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()
        for m in [self.alpha_b2b, self.alpha_u2b, self.alpha_u2u, self.alpha_b2u]:
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos_edge_index: LongTensor,
        neg_edge_index: LongTensor,
        pos_weight: Tensor,
        neg_weight: Tensor
    ):
        pos_edges_no_self, pos_w_no_self = remove_self_loops(pos_edge_index, edge_attr=pos_weight)
        neg_edges, neg_w = remove_self_loops(neg_edge_index, edge_attr=neg_weight)

        if self.add_self_loops:
            pos_edges, pos_w = add_self_loops(
                pos_edges_no_self, edge_attr=pos_w_no_self, fill_value=1.0
            )
        else:
            pos_edges, pos_w = pos_edges_no_self, pos_w_no_self

        if self.first_aggr:
            emb_balanced = self.lin_b(x)
            emb_unbalanced = self.lin_u(x)
            out_balanced = self.propagate(
                pos_edges,
                x1=emb_balanced, x2=emb_balanced,
                edge_p=torch.zeros(pos_edges.size(1), dtype=torch.long, device=pos_edges.device),
                edge_attr=pos_w,
                target_is_B=True,
            )
            out_unbalanced = self.propagate(
                neg_edges,
                x1=emb_unbalanced, x2=emb_unbalanced,
                edge_p=torch.zeros(neg_edges.size(1), dtype=torch.long, device=neg_edges.device),
                edge_attr=neg_w,
                target_is_B=False,
            )
            return torch.cat([out_balanced, out_unbalanced], dim=-1)

        d = self.in_dim
        emb_balanced = x[..., :d]
        emb_unbalanced = x[..., d:]
        merged_edges = torch.cat([pos_edges, neg_edges], dim=1)
        merged_w = torch.cat([pos_w, neg_w], dim=0)
        merged_p = torch.cat([
            torch.zeros(pos_edges.size(1), dtype=torch.long, device=merged_edges.device),
            torch.ones(neg_edges.size(1), dtype=torch.long, device=merged_edges.device)
        ], dim=0)
        
        assert merged_edges.size(1) == merged_w.numel() == merged_p.numel()
        Epos = pos_edges.size(1)
        assert torch.all(merged_p[:Epos] == 0)
        assert torch.all(merged_p[Epos:] == 1)
        
        out_balanced = self.propagate(
            merged_edges,
            x1=self.lin_b(emb_balanced),
            x2=self.lin_b(emb_unbalanced),
            edge_p=merged_p,
            edge_attr=merged_w,
            target_is_B=True,
        )

        out_unbalanced = self.propagate(
            merged_edges,
            x1=self.lin_u(emb_unbalanced),
            x2=self.lin_u(emb_balanced),
            edge_p=merged_p,
            edge_attr=merged_w,
            target_is_B=False,
        )
        return torch.cat([out_balanced, out_unbalanced], dim=-1)

    def message(
        self,
        x1_j: Tensor, x2_j: Tensor,
        x1_i: Tensor, x2_i: Tensor,
        edge_p: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        edge_attr: Tensor,
        target_is_B: bool,
    ) -> Tensor:
        f1 = torch.cat([x1_j, x1_i], dim=-1)
        f2 = torch.cat([x2_j, x2_i], dim=-1)

        edge_p = edge_p.to(device=f1.device, dtype=torch.long)
        row = torch.arange(f1.size(0), device=f1.device)

        edge_h = torch.stack([f1, f2], dim=-1)
        edge_h = edge_h[row, :, edge_p]

        if target_is_B:
            e0 = self.alpha_b2b(edge_h)
            e1 = self.alpha_u2b(edge_h)
        else:
            e0 = self.alpha_u2u(edge_h)
            e1 = self.alpha_b2u(edge_h)

        e = torch.where(edge_p.view(-1, 1) == 0, e0, e1)
        alpha = softmax(torch.tanh(e), index, ptr, size_i)

        x_j = torch.stack([x1_j, x2_j], dim=-1)
        x_j = x_j[row, :, edge_p]
        return x_j * alpha * edge_attr.view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_dim}, {self.out_dim}, first_aggr={self.first_aggr})'


class Piglet(nn.Module):
    def __init__(
        self,
        node_num: int,
        in_dim: int = 64,
        out_dim: int = 64,
        lamb: int = 1,
        device='cpu',
        layer_num: int = 2,
        init_emb: torch.FloatTensor = None,
        init_emb_grad: bool = False,
    ):
        super().__init__()

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb
        self.device = device

        self.x = nn.Parameter(init_emb, requires_grad=init_emb_grad)

        self.conv1 = PigletConv(in_dim, out_dim // 2, first_aggr=True)
        self.convs = nn.ModuleList([
            PigletConv(out_dim // 2, out_dim // 2, first_aggr=False)
            for _ in range(layer_num - 1)
        ])
        self.weight = nn.Linear(self.out_dim, self.out_dim)
        self.structure_loss = Sign_Structure_Loss()
        
        hid = out_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * out_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        ).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.conv1, "reset_parameters"):
            self.conv1.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        self.weight.reset_parameters()
        for m in self.edge_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def loss(self, z, pos_edge_index, neg_edge_index) -> torch.FloatTensor:
        structure_loss = self.structure_loss(z, pos_edge_index, neg_edge_index)
        return self.lamb * structure_loss

    def forward(self, pos_edge_index, neg_edge_index, pos_weight, neg_weight) -> Tensor:
        z = torch.tanh(self.conv1(
            self.x, pos_edge_index, neg_edge_index, pos_weight, neg_weight))
        for conv in self.convs:
            z = torch.tanh(conv(z, pos_edge_index, neg_edge_index, pos_weight, neg_weight))
        z = torch.tanh(self.weight(z))
        return z