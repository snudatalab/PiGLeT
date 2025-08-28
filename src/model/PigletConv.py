from typing import Optional, Union

from torch_geometric.typing import (PairTensor, OptTensor)
import torch
from torch import LongTensor, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (add_self_loops,
                                   softmax,
                                   remove_self_loops)


class PigletConv(MessagePassing):
  
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = True,
        add_self_loops=True,
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

        self.alpha_u = torch.nn.Linear(self.out_dim * 2, 1)
        self.alpha_b = torch.nn.Linear(self.out_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()
        torch.nn.init.xavier_normal_(self.alpha_b.weight)
        torch.nn.init.xavier_normal_(self.alpha_u.weight)

        self.alpha_b.reset_parameters()
        self.alpha_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: LongTensor,
            neg_edge_index: LongTensor, pos_weight, neg_weight):
        """"""
        orig_pos_weight = pos_weight
        orig_neg_weight = neg_weight

        if self.first_aggr:
            emb_balanced = self.lin_b(x)
            emb_unbalanced = self.lin_u(x)

            pos_edges, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges, pos_weights = add_self_loops(pos_edges, edge_attr=pos_weights_no_self, fill_value=1.0)

            edge_types = torch.zeros(pos_edges.size(-1), dtype=torch.long)

            out_balanced = self.propagate(
                pos_edges, x1=emb_balanced, x2=emb_balanced,
                edge_p=edge_types, alpha_func=self.alpha_b, edge_attr=pos_weights
            )

            neg_edges, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=orig_neg_weight)
            neg_edges, neg_weights = add_self_loops(neg_edges, edge_attr=neg_weights_no_self, fill_value=1.0)

            edge_types = torch.zeros(neg_edges.size(-1), dtype=torch.long)

            out_unbalanced = self.propagate(
                neg_edges, x1=emb_unbalanced, x2=emb_unbalanced,
                edge_p=edge_types, alpha_func=self.alpha_u, edge_attr=neg_weights
            )

            combined_out = torch.cat([out_balanced, out_unbalanced], dim=-1)

        else:
            feature_dim = self.in_dim
            emb_balanced = x[..., :feature_dim]
            emb_unbalanced = x[..., feature_dim:]

            pos_edges_no_self, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges_with_self, pos_weights_with_self = add_self_loops(pos_edges_no_self, edge_attr=pos_weights_no_self, fill_value=1.0)
            neg_edges_no_self, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=neg_weight)

            merged_edges = torch.cat([pos_edges_with_self, neg_edges_no_self], dim=-1)
            merged_weights = torch.cat([pos_weights_with_self, neg_weights_no_self], dim=-1)

            edge_labels_pos = torch.zeros(pos_edges_with_self.size(-1), dtype=torch.long)
            edge_labels_neg = torch.ones(neg_edges_no_self.size(-1), dtype=torch.long)
            merged_edge_labels = torch.cat([edge_labels_pos, edge_labels_neg], dim=-1)

            transformed_balanced_x1 = self.lin_b(emb_balanced)
            transformed_balanced_x2 = self.lin_b(emb_unbalanced)

            out_balanced = self.propagate(
                merged_edges, x1=transformed_balanced_x1, x2=transformed_balanced_x2,
                edge_p=merged_edge_labels, alpha_func=self.alpha_b, edge_attr=merged_weights
            )

            pos_edges_no_self, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges_with_self, pos_weights_with_self = add_self_loops(pos_edges_no_self, edge_attr=pos_weights_no_self, fill_value=1.0)
            neg_edges_no_self, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=orig_neg_weight)

            merged_edges = torch.cat([pos_edges_with_self, neg_edges_no_self], dim=-1)
            merged_weights = torch.cat([neg_weights_no_self, pos_weights_with_self], dim=-1)

            edge_labels_pos = torch.zeros(pos_edges_with_self.size(-1), dtype=torch.long)
            edge_labels_neg = torch.ones(neg_edges_no_self.size(-1), dtype=torch.long)
            merged_edge_labels = torch.cat([edge_labels_pos, edge_labels_neg], dim=-1)

            transformed_unbalanced_x1 = self.lin_u(emb_unbalanced)
            transformed_unbalanced_x2 = self.lin_u(emb_balanced)

            out_unbalanced = self.propagate(
                merged_edges, x1=transformed_unbalanced_x1, x2=transformed_unbalanced_x2,
                edge_p=merged_edge_labels, alpha_func=self.alpha_u, edge_attr=merged_weights
            )

            combined_out = torch.cat([out_balanced, out_unbalanced], dim=-1)

        return combined_out


    def message(self, x1_j: Tensor, x2_j: Tensor, x1_i: Tensor, x2_i: Tensor, edge_p: Tensor, alpha_func, index: Tensor, ptr: OptTensor,
        size_i: Optional[int], edge_attr: Tensor) -> Tensor:
        x1 = torch.cat([x1_j, x1_i], dim=-1)
        x2 = torch.cat([x2_j, x2_i], dim=-1)
        edge_h = torch.stack([x1, x2], dim=-1)
        edge_h = edge_h[torch.arange(edge_h.size(0)), :, edge_p]

        alpha = alpha_func(edge_h)
        alpha = torch.tanh(alpha)
        alpha = softmax(alpha, index, ptr, size_i)

        x_i = torch.stack([x1_i, x2_i], dim=-1)
        x_i = x_i[torch.arange(edge_h.size(0)), :, edge_p]
        return x_i * alpha * edge_attr.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


