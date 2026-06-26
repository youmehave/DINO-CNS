"""
============================================================================
  GraphVS_Transformer — 时序 Transformer 记忆替代 GRU 的消融实验
============================================================================
  基于原始 GraphVS (cns/models/graph_vs.py)，将 PEConvGRUCell 替换为基于
  MultiheadAttention 的时序记忆检索模块 TransformerTemporalAggr。

  【零侵入设计】
  - 前向接口与 GraphVS 完全一致: forward(data, hidden) → (vec, norm, hidden)
  - hidden 语义兼容: None=新场景, Tensor=历史记忆（形状从 [N,D] 变为 [T,N,D]）
  - BPTT 截断兼容: clone().detach() 对序列张量同样有效
  - 训练/推理代码无需任何修改

  【架构变更】
  原始:  PERConv → PEConvGRUCell → PERConv
  修改:  PERConv → TransformerTemporalAggr (MHA+FFN+滑动窗口记忆) → PERConv

  作者: Claude (基于 GraphVS 原始代码)
  日期: 2025-06-26
============================================================================
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# [原始导入] — 复用 GraphVS 中不变的组件
# ============================================================================
from cns.models.graph_vs import (
    MLP, Encoder, Decoder, PERConv, GraphVS, postprocess, objectives
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  新增模块 1: TransformerTemporalAggr                                    ║
# ║  替代原始 PEConvGRUCell，提供基于自注意力的时序记忆检索                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TransformerTemporalAggr(nn.Module):
    """
    [新增] 时序 Transformer 聚合器

    功能: 维护最近 max_len 帧的特征序列作为显式记忆库，用当前帧特征作为 Query
          对历史帧进行交叉注意力检索，通过 FFN + 残差连接融合时序信息。

    与 PEConvGRUCell 的接口差异:
      PEConvGRUCell:  forward(h, x, pos, edge_index_gate, edge_index_cand) → h_next
      Transformer:    forward(hidden_seq, x_clu) → (out, next_hidden_seq)

    参数:
        hidden_dim: 特征维度
        num_heads:  多头注意力头数 (默认 4)
        max_len:    滑动窗口最大记忆帧数 (默认 8)
        dropout:    attention dropout 比率 (默认 0.0)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 max_len: int = 8, dropout: float = 0.0):
        super(TransformerTemporalAggr, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_heads = num_heads

        # --- 多头交叉注意力 ---
        # 当前帧 Query 去检索历史帧 Key/Value
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # [Batch, Seq, Dim] 格式
        )

        # --- 可学习时序位置编码 ---
        # 位置 0 = 最早帧, 位置 max_len-1 = 最新帧
        # Shape: [1, max_len, hidden_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # --- 归一化层 ---
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # --- 前馈网络 (FFN) ---
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_seq: torch.Tensor, x_clu: torch.Tensor):
        """
        参数:
            hidden_seq: 历史特征序列 [SeqLen, NumClusters, HiddenDim]
                        若为 None，表示新场景首帧
            x_clu:      当前帧聚类特征 [NumClusters, HiddenDim]

        返回:
            out:             时序融合后的特征 [NumClusters, HiddenDim]
            next_hidden_seq: 更新后的记忆序列 [NewSeqLen, NumClusters, HiddenDim]
        """
        # ---- 1. 更新滑动窗口记忆库 ----
        if hidden_seq is None:
            # 首帧：创建新的单帧记忆
            next_hidden_seq = x_clu.unsqueeze(0)  # [1, N, D]
        else:
            # 拼接当前帧并滑动窗口截断
            next_hidden_seq = torch.cat(
                [hidden_seq, x_clu.unsqueeze(0)], dim=0
            )  # [SeqLen+1, N, D]
            if next_hidden_seq.size(0) > self.max_len:
                next_hidden_seq = next_hidden_seq[-self.max_len:]

        seq_len = next_hidden_seq.size(0)

        # ---- 2. 维度变换适配 PyTorch MHA ----
        # MHA 的 batch_first 格式: [Batch, Seq, Dim]
        # 这里: Batch = NumClusters, Seq = 记忆帧数
        kv = next_hidden_seq.permute(1, 0, 2)  # [N, SeqLen, D]

        # 叠加位置编码（对齐到最近的 max_len 个位置）
        pos = self.pos_embed[:, -seq_len:, :]  # [1, SeqLen, D]
        kv_pos = kv + pos                       # [N, SeqLen, D]

        # Query: 当前帧特征 [N, 1, D]
        q = x_clu.unsqueeze(1)                  # [N, 1, D]

        # ---- 3. 时序交叉注意力 ----
        # 当前帧 (Query) 关注所有历史帧 (Key/Value)
        attn_out, attn_weights = self.mha(
            query=q,
            key=kv_pos,
            value=kv,
            need_weights=False
        )  # attn_out: [N, 1, D]
        attn_out = attn_out.squeeze(1)          # [N, D]

        # ---- 4. 残差连接 + LayerNorm ----
        x_updated = self.norm1(x_clu + attn_out)

        # ---- 5. FFN + 最终输出 ----
        out = self.norm2(x_updated + self.ffn(x_updated))

        return out, next_hidden_seq


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  修改模块 2: Backbone                                                   ║
# ║  与原始 GraphVS.Backbone 唯一差异: temporal_aggr 使用 Transformer        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Backbone(nn.Module):
    """
    [修改] 主干网络 — 时序模块从 PEConvGRUCell 替换为 TransformerTemporalAggr

    原始代码位置: cns/models/graph_vs.py  line 234-251
    变更标注:     见下方 <<< 修改 >>> 标记

    接口保持与原始 Backbone 完全一致:
        forward(hidden, x_clu, pos_clu, l1_dense_edge_index_cur,
                l1_dense_edge_index_tar, batch_clu) → (hidden, x_clu)
    """

    def __init__(self, hidden_dim, pos_dim,
                 # [新增参数] Transformer 配置
                 num_heads=4, max_len=8, dropout=0.0):
        super(Backbone, self).__init__()

        # === 原始组件 (未修改) ===
        self.l1_conv0 = PERConv(hidden_dim, hidden_dim, pos_dim)
        self.l1_conv1 = PERConv(hidden_dim, hidden_dim, pos_dim)

        # === <<< 修改 >>> 时序聚合器: PEConvGRUCell → TransformerTemporalAggr ===
        #
        # 原始代码:
        #   self.temporal_aggr = PEConvGRUCell(hidden_dim, hidden_dim, pos_dim)
        #
        # 修改后:
        self.temporal_aggr = TransformerTemporalAggr(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        # =========================================================================

    def forward(self, hidden, x_clu, pos_clu,
                l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu):
        """
        [修改] 前向传播

        与原始唯一的差异: temporal_aggr 调用不使用 pos 和 edge_index 参数，
        因为 Transformer 只做纯时序检索（空间建模由两侧的 PERConv 完成）。

        原始代码:
            hidden = self.temporal_aggr(
                hidden, x_clu, pos_clu,
                edge_index_gate=l1_dense_edge_index_tar,
                edge_index_cand=l1_dense_edge_index_cur
            )

        修改后:
            x_clu_aggr, hidden = self.temporal_aggr(hidden, x_clu)
        """
        # 第一个 PERConv：空间图卷积（L1 簇间图）
        x_clu = F.relu(self.l1_conv0(x_clu, pos_clu, l1_dense_edge_index_cur, batch_clu))

        # === <<< 修改 >>> 时序聚合 ===
        # 原始: hidden = self.temporal_aggr(hidden, x_clu, pos_clu,
        #         edge_index_gate=l1_dense_edge_index_tar, edge_index_cand=l1_dense_edge_index_cur)
        # 修改: 纯时序检索，不需要空间参数
        x_clu_aggr, hidden = self.temporal_aggr(hidden, x_clu)
        # =============================

        # 第二个 PERConv：空间图卷积（对时序融合后的特征再做空间聚合）
        x_clu = F.relu(self.l1_conv1(x_clu_aggr, pos_clu, l1_dense_edge_index_cur, batch_clu))

        return hidden, x_clu


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  修改模块 3: GraphVS_Transformer                                        ║
# ║  继承 GraphVS，仅替换 Backbone                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class GraphVS_Transformer(GraphVS):
    """
    [修改] GraphVS 的 Transformer 变体

    继承原始 GraphVS 的 Encoder / Decoder / postprocess / objectives，
    仅替换 Backbone 为时序 Transformer 版本。

    与 GraphVS 唯一的差异:
      1. Backbone 使用 Transformer 时序聚合
      2. init_hidden() 返回 None（由 TransformerTemporalAggr 内部处理）

    所有外部接口保持一致:
      - forward(data, hidden) → (vel_si_vec, vel_si_norm, hidden)
      - preprocess(data) → data
      - postprocess(raw_pred, data) → vel
      - objectives(raw_pred, data) → (result, loss)
      - get_parameter_groups() → (pg_wi_decay, pg_wo_decay)
    """

    def __init__(self, input_dim, pos_dim, hidden_dim=128, regress_norm=True,
                 # [新增参数] Transformer 配置（仅对此变体有效）
                 num_heads=4, max_len=8, dropout=0.0):
        """
        参数:
            input_dim:     输入特征维度
            pos_dim:       位置特征维度
            hidden_dim:    隐藏层维度 (默认 128)
            regress_norm:  是否回归范数 (默认 True)
            num_heads:     [新增] MHA 注意力头数
            max_len:       [新增] 滑动窗口记忆帧数
            dropout:       [新增] attention + FFN dropout
        """
        super(GraphVS_Transformer, self).__init__(
            input_dim, pos_dim, hidden_dim, regress_norm
        )

        # === <<< 修改 >>> 删除原始 Backbone，替换为 Transformer 版本 ===
        # 原始: self.backbone = Backbone(hidden_dim, pos_dim)
        del self.backbone
        self.backbone = Backbone(
            hidden_dim=hidden_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        # ==============================================================

    def init_hidden(self, size0):
        """
        [修改] 初始化隐藏状态

        原始代码:
            return self.backbone.temporal_aggr.init_hidden(size0)
            → 返回 torch.zeros(size0, hidden_dim)

        修改后:
            Transformer 的记忆库在第一个前向传播时自动创建，
            初始状态为 None（表示空记忆）。
        """
        # === <<< 修改 >>>
        # 原始: return self.backbone.temporal_aggr.init_hidden(size0)
        return None
        # =================

    def forward(self, data, hidden=None):
        """
        [修改] 前向传播 — 覆盖父类 GraphVS.forward

        原始 GraphVS.forward 在 hidden=None 时调用:
            hidden = self.init_hidden(...).to(x_cur)
        由于 Transformer 版本的 init_hidden 返回 None,
        直接跳过该初始化，将 None 传递给 Backbone。

        其余逻辑与原始 GraphVS.forward 完全一致。
        """
        # === 从 GraphVS.forward 复用 (未修改) ===
        x_cur = getattr(data, "x_cur")
        x_tar = getattr(data, "x_tar")
        pos_cur = getattr(data, "pos_cur")
        pos_tar = getattr(data, "pos_tar")

        l1_dense_edge_index_cur = getattr(data, "l1_dense_edge_index_cur")
        l1_dense_edge_index_tar = getattr(data, "l1_dense_edge_index_tar")

        l0_to_l1_edge_index_j_cur = getattr(data, "l0_to_l1_edge_index_j_cur")
        l0_to_l1_edge_index_i_cur = getattr(data, "l0_to_l1_edge_index_i_cur")
        l0_to_l1_edge_index_cur = torch.stack(
            [l0_to_l1_edge_index_j_cur, l0_to_l1_edge_index_i_cur], dim=0
        )

        cluster_mask = getattr(data, "cluster_mask")
        cluster_centers_index = getattr(data, "cluster_centers_index")

        if not hasattr(data, "batch"):
            batch = None
        else:
            batch = getattr(data, "batch")
        if batch is None:
            batch = torch.zeros(x_cur.size(0)).long().to(x_cur.device)

        x_clu = self.encoder(
            x_cur, x_tar, pos_cur, pos_tar, cluster_mask,
            l0_to_l1_edge_index_cur, cluster_centers_index
        )
        pos_clu = pos_tar[cluster_centers_index]
        batch_clu = batch[cluster_centers_index]

        # === <<< 修改 >>> 隐藏状态初始化 ===
        # 原始代码:
        #   if hidden is None:
        #       hidden = self.init_hidden(getattr(data, "num_clusters").sum()).to(x_cur)
        #
        # 修改后:
        #   Transformer 接受 None 作为有效的初始状态（表示空记忆），
        #   在 Backbone 内部由 TransformerTemporalAggr 自动创建首帧记忆。
        #   因此跳过 init_hidden 调用，直接传 None。
        # =====================================
        hidden, x_clu = self.backbone(
            hidden, x_clu, pos_clu,
            l1_dense_edge_index_cur, l1_dense_edge_index_tar, batch_clu
        )

        vel_si_vec, vel_si_norm = self.decoder(x_clu, cluster_mask, batch_clu)

        return vel_si_vec, vel_si_norm, hidden


# ============================================================================
# [测试入口] — 与原始 graph_vs.py 保持一致的测试逻辑
# ============================================================================

if __name__ == "__main__":
    from cns.sim.dataset import DataLoader

    dataloader = DataLoader(
        None,
        batch_size=2, train=True, num_trajs=100,
        env="Point"
    )

    net = GraphVS_Transformer(
        input_dim=dataloader.num_node_features,
        pos_dim=dataloader.num_pos_features,
        hidden_dim=128,
        regress_norm=True,
        num_heads=4,
        max_len=8
    )

    print(f"[INFO] GraphVS_Transformer created successfully")
    print(f"[INFO] Parameters: {sum(p.numel() for p in net.parameters()):,}")

    hidden = None
    for i, data in enumerate(dataloader):
        if getattr(data, "new_scene").any():
            hidden = None
            print(f"\n[INFO] Step {i}: New scene, reset hidden state")

        raw_pred = net(data, hidden)
        vel_si_vec, vel_si_norm, hidden = raw_pred

        vel = net.postprocess(raw_pred, data)
        dataloader.feedback(data.vel)

        print(f"  Step {i}: vel={vel.squeeze().detach().numpy().round(3)}, "
              f"mem_len={hidden.size(0) if hidden is not None else 0}")

        if i >= 5:
            break

    print("\n[INFO] GraphVS_Transformer smoke test passed ✓")
