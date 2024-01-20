import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.meta import MetaLayer
import torch.nn as nn
import inspect
import yaml
import sys
import deepgate as dg
import numpy as np
from .utils import one_hot

_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid
}

graph_emb_dict = {}


class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=2, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5, sigmoid=False, tanh=False):
        super(MLP, self).__init__()

        assert num_layer >= 2, "The number of layers should be larger or equal to 2."
        if norm_layer in _norm_layer_factory.keys():
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory.keys():
            self.act_layer = _act_layer_factory[act_layer]

        if p_drop > 0:
            self.dropout = nn.Dropout

        fc = []
        # 1st layer
        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))

        # last layer
        fc.append(nn.Linear(dim_hidden, dim_pred))
        # sigmoid
        if sigmoid:
            fc.append(nn.Sigmoid())
        if tanh:
            fc.append(nn.Sigmoid())

        self.fc = nn.Sequential(*fc)


    def forward(self, x):
        out = self.fc(x)
        return out

class ckt_net(nn.Module):
    def __init__(self, args):
        super(ckt_net, self).__init__()

        self.ckt_model = dg.Model().to(args.device)

        self.ckt_model.load_pretrained()

        self.mlp = MLP(dim_in=256, dim_hidden=32, dim_pred=2, 
                       num_layer=6, p_drop=0.2, act_layer='relu').to(args.device)

        self.device = args.device

    def forward_one(self, obs):
        obs = obs.to(self.device)

        if obs.name not in graph_emb_dict:
            hs, hf = self.ckt_model(obs)
            graph_emb = torch.cat([hs, hf], dim=1)
            graph_emb_dict[obs.name] = graph_emb
        else:
            graph_emb = graph_emb_dict[obs.name]
            
        # Stone / 01-20: only infer the valid var (var_state == 2)
        pred_mask = self.mlp(graph_emb[obs.valid_mask])
        y_pred = pred_mask.reshape(-1)

        # y_pred = self.mlp(graph_emb)
        # valid_mask = obs.valid_mask

        return y_pred

    def forward_batch(self, obs):
        batch_size = len(obs)

        for i in range(batch_size):
            obs[i] = obs[i].to(self.device)

        # x = torch.cat([obs[i].x for i in range(batch_size)], dim=0)
        # edge_index = torch.cat([obs[i].edge_index for i in range(batch_size)], dim=1)
        # forward_level = torch.cat([obs[i].forward_level for i in range(batch_size)], dim=0)
        # backward_level = torch.cat([obs[i].backward_level for i in range(batch_size)], dim=0)
        # forward_index = torch.cat([obs[i].forward_index for i in range(batch_size)], dim=0)
        # backward_index = torch.cat([obs[i].backward_index for i in range(batch_size)], dim=0)
        # gate = torch.cat([obs[i].gate for i in range(batch_size)], dim=0)
        # PIs = torch.cat([obs[i].PIs for i in range(batch_size)], dim=0)
        # POs = torch.cat([obs[i].POs for i in range(batch_size)], dim=0)
        # graph = dg.OrderedData(edge_index=edge_index, x=x, forward_level=forward_level, backward_level=backward_level,
        #                     forward_index=forward_index, backward_index=backward_index)
        # graph.gate = gate
        # graph.PIs = PIs
        # graph.POs = POs

        batch_graph_emb = torch.empty(0, 256).to(self.device)

        for i in range(batch_size):
            if obs[i].name not in graph_emb_dict:
                # Merge
                hs, hf = self.ckt_model(obs[i])
                graph_emb = torch.cat([hs, hf], dim=1)
                graph_emb_dict[obs[i].name] = graph_emb
            else:
                graph_emb = graph_emb_dict[obs[i].name]
            batch_graph_emb = torch.cat((batch_graph_emb, graph_emb), dim=0)
        
        batch_graph_emb = batch_graph_emb.detach()
        
        # number of nodes
        node_num = torch.tensor([torch.tensor([obs[i].x.shape[0]]) for i in range(batch_size)])
        gather_index = node_num.cumsum(0).roll(1)
        gather_index[0] = 0
        valid_mask = torch.cat([torch.tensor(obs[i].valid_mask) + gather_index[i] for i in range(batch_size)], dim=0)

        # forward
        y_pred = self.mlp(batch_graph_emb[valid_mask])
        
        return y_pred

    def forward(self, obs):
        # hs: structural embeddings, hf: functional embeddings
        # hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
        if isinstance(obs, list) or isinstance(obs, np.ndarray):
            return self.forward_batch(obs)
        else:
            return self.forward_one(obs)

    def save(self, path):
        torch.save(self.state_dict(), path)