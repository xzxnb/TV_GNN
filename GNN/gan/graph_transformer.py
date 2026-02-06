import pytorch_lightning as pl
import torch_geometric as tg
import torch
from nets.SBMs_node_classification import graph_transformer_net
import json
import numpy as np
import dgl
from scipy import sparse as sp
import hashlib
import pdb
class GF(pl.LightningModule):
    def __init__(
        self,
        n_node_classes: int,
        n_edge_classes: int,
        domain_size: int,
        allow_floats_adj: bool = False,
        val_metrics_file: str = None,
    ):
        super().__init__()
        self.allow_floats_adj = allow_floats_adj
        self.val_metrics_file = val_metrics_file
        self.domain_size = domain_size
        # 合并默认 net_params（允许调用方只传递覆盖项）
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_params = {
            # 基本尺寸从本模块入参推断
            "in_dim": n_node_classes+1,
            "hidden_dim": 64,
            "out_dim": 32,
            "n_classes": 8,
            # Transformer 超参数
            "n_heads": 8,
            "in_feat_dropout": 0.3,
            "dropout": 0.1,
            "L": 8,
            # 规范化/残差与读出
            "readout": "mean",
            "layer_norm": True,
            "batch_norm": True,
            "residual": True,
            # 设备与位置编码
            "device": device_str,
            "lap_pos_enc": True,
            "wl_pos_enc": False,
            # 仅当 lap_pos_enc=True 时才会使用
            "pos_enc_dim": domain_size-2,
        }
        self.layer = graph_transformer_net.GraphTransformerNet(self.net_params)
        self.aggregator = torch.nn.Linear(self.net_params["n_classes"], 1)
        self.classifier = torch.nn.Linear(domain_size * n_edge_classes, 1)


    def process_subgraph(self, node_features, adj_matrix):
        assert node_features.dim() == 3 and node_features.size(1) == adj_matrix.size(
            1
        ), f"{node_features.shape=}, {adj_matrix.shape=}, [batch_size, n_nodes, n_feat]"
        assert adj_matrix.dim() == 3 and adj_matrix.size(1) == adj_matrix.size(
            2
        ), f"{adj_matrix.shape=}, [batch_size, n_nodes, n_nodes]"
        assert self.allow_floats_adj or bool(
            ((adj_matrix == 0) | (adj_matrix == 1)).all()
        ), "Adj matrix should be zeros and ones"

        # dgl库
        all_x = []
        for i in range(adj_matrix.shape[0]):
            adj_t = adj_matrix[i].T
            sa = torch.allclose(adj_matrix[i], adj_t, atol=1e-6)
            # if not sa:
            #     pdb.post_mortem()
            u, v = torch.where(adj_matrix[i] > 0)
            g = dgl.graph((u, v), num_nodes=self.domain_size)
            g = laplacian_positional_encoding(g, self.net_params["pos_enc_dim"])
            g = wl_positional_encoding(g)
            try:
                batch_lap_pos_enc = g.ndata['lap_pos_enc']
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(g.device)
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
            except Exception as e:
                print(f"捕获到异常: {type(e).__name__}, 信息: {str(e)}")
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = g.ndata['wl_pos_enc']
            except:
                batch_wl_pos_enc = None
            x = self.layer(g, node_features[i], batch_lap_pos_enc, batch_wl_pos_enc)
            # x = self.aggregator(x).squeeze(-1)
            x = x.squeeze(-1)
            # x = torch.nn.functional.tanh(x)
            all_x.append(x)
        return torch.stack(all_x, dim=0)

    def forward(self, node_features, edge_type_to_adj_matrix):
        subgraphs = [
            self.process_subgraph(node_features, adj_matrix)
            for adj_matrix in edge_type_to_adj_matrix.values()
        ]
        x = torch.cat(subgraphs, dim=-1)

        x = x.reshape(-1)

        x = torch.nn.functional.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.node_feats, batch.edge_type_to_adj_matrix)
        loss = torch.nn.functional.binary_cross_entropy(out, batch.gan_y)
        # self.log("train_loss", loss, batch_size=len(batch.gan_y))
        # self.log(
        #     "train_accuracy", accuracy(out, batch.gan_y), batch_size=len(batch.gan_y)
        # )
        # self.log(
        #     "train_precision", precision(out, batch.gan_y), batch_size=len(batch.gan_y)
        # )
        # self.log("train_recall", recall(out, batch.gan_y), batch_size=len(batch.gan_y))
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.node_feats, batch.edge_type_to_adj_matrix)
        loss = torch.nn.functional.binary_cross_entropy(out, batch.gan_y)
        self.log("val_loss", loss, batch_size=len(batch.gan_y))
        val_accuracy = accuracy(out, batch.gan_y)
        val_precision = precision(out, batch.gan_y)
        val_recall = recall(out, batch.gan_y)
        self.log(
            "val_accuracy", val_accuracy, prog_bar=True, batch_size=len(batch.gan_y)
        )
        self.log("val_precision", val_precision, batch_size=len(batch.gan_y))
        self.log("val_recall", val_recall, batch_size=len(batch.gan_y))
        return {
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
        }

    def validation_epoch_end(self, outputs):
        val_accuracy = sum(x["val_accuracy"] for x in outputs) / len(outputs)
        val_precision = sum(x["val_precision"] for x in outputs) / len(outputs)
        val_recall = sum(x["val_recall"] for x in outputs) / len(outputs)

        # Initialize highest values if they don't exist
        if not hasattr(self, "highest_val_accuracy"):
            self.highest_val_accuracy = 0.0
            self.highest_val_precision = 0.0
            self.highest_val_recall = 0.0

        # Update highest values
        self.highest_val_accuracy = max(self.highest_val_accuracy, val_accuracy)
        self.highest_val_precision = max(self.highest_val_precision, val_precision)
        self.highest_val_recall = max(self.highest_val_recall, val_recall)

        # Save the highest values
        with open(self.val_metrics_file, "w") as f:
            json.dump(
                {
                    "val_accuracy": self.highest_val_accuracy,
                    "val_precision": self.highest_val_precision,
                    "val_recall": self.highest_val_recall,
                },
                f,
                indent=2,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_hard = (pred > 0.5).float()
    return (pred_hard == target).float().mean().item()


def precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_hard = (pred > 0.5).float()
    true_positives = ((pred_hard == 1) & (target == 1)).float().sum().item()
    predicted_positives = (pred_hard == 1).float().sum().item()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0


def recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_hard = (pred > 0.5).float()
    true_positives = ((pred_hard == 1) & (target == 1)).float().sum().item()
    actual_positives = (target == 1).float().sum().item()
    return true_positives / actual_positives if actual_positives > 0 else 0.0


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().to(g.device)

    return g


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding
        adapted from

        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """

    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().cpu().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values())).to(g.device)
    return g