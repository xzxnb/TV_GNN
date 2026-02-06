import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer

"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        # 获取消息的维度（假设为[num_edges, num_heads, feat_dim]）
        num_heads = g.ndata['V_h'].shape[1]
        feat_dim = g.ndata['V_h'].shape[2]

        # 动态初始化wV列（目标节点数 × 头数 × 特征维）
        g.dstdata['wV'] = torch.zeros(
            (g.number_of_dst_nodes(), num_heads, feat_dim),
            device=g.device
        )

        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

        # has_nan_wV = torch.isnan(g.ndata['wV']).any().item()
        # has_zero_z = (g.ndata['z'] == 0).any().item()
        # if has_zero_z == 1 or has_nan_wV == 1:
        #     pdb.set_trace()
        # 输出结果
        # print(f"wV 中存在 NaN: {has_nan_wV}")
        # print(f"z 中存在 0: {has_zero_z}")

    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        # print("V_h维度:", g.srcdata['V_h'].shape)  # 应为 (N, d_v)，如(8,4)
        # print("K_h维度:", g.srcdata['K_h'].shape)  # 应为 (N, d_k)，如(8,4)
        # print("Q_h维度:", g.dstdata['Q_h'].shape)  # 应为 (N, d_k)，如(8,4)

        self.propagate_attention(g)
        # has_nan_wV = torch.isnan(g.ndata['wV']).any().item()
        # has_zero_z = (g.ndata['z'] == 0).any().item()
        # if has_zero_z==1 or has_nan_wV==1:
        #     pdb.set_trace()
        # 输出结果
        # print(f"wV 中存在 NaN: {has_nan_wV}")
        # print(f"z 中存在 0: {has_zero_z}")
        isolated_nodes = (g.out_degrees(g.nodes()) == 0) | (g.in_degrees(g.nodes()) == 0)
        g.ndata['z'][isolated_nodes] = torch.ones_like(g.ndata['z'][isolated_nodes])

        head_out = g.ndata['wV'] / g.ndata['z']
        g.ndata['wV'] = g.ndata['V_h'] + g.ndata['wV']

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O = nn.Linear(out_dim, out_dim)
        self.READ = nn.Linear(out_dim, out_dim)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        # hg = hg.unsqueeze(0)
        hg = hg.repeat(h.shape[0], 1)
        hg = self.READ(hg)

        has_nan = torch.isnan(h).any()
        if has_nan == 1:
            pdb.set_trace()

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        if self.residual:
            h = h_in1 + h + hg  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)