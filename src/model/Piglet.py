import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric_signed_directed.utils.signed import (create_spectral_features,
                                                          Link_Sign_Entropy_Loss, Sign_Structure_Loss)


from .PigletConv import PigletConv

class Piglet(nn.Module):
    def __init__(
        self,
        node_num: int,
        in_dim: int = 64,
        out_dim: int = 64,
        lamb : int =1,
        device= 'cpu',
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


        self.conv1 = PigletConv(in_dim, out_dim // 2,
                              first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(
                PigletConv(out_dim // 2, out_dim // 2,
                         first_aggr=False))

        self.weight = torch.nn.Linear(self.out_dim, self.out_dim)
        self.structure_loss = Sign_Structure_Loss()

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()
        nn.init.xavier_uniform_(self.edge_classifier.weight)
        if self.edge_classifier.bias is not None:
            nn.init.zeros_(self.edge_classifier.bias)
        
    
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
