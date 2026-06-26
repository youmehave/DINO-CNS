"""
============================================================================
  train_graph_vs_transformer.py
  GraphVS_Transformer 训练入口
============================================================================
  用法:
    python -m cns.ablation.temporal_transformer.train_graph_vs_transformer

  与原始 train_cns.py 的唯一差异:
    - model_class=GraphVS_Transformer (而非 GraphVS)
    - 额外的 num_heads / max_len / dropout 参数

  所有训练逻辑（BPTT截断、Teacher Forcing、学习率调度）保持不变。
============================================================================
"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import argparse
import torch
import torch.optim as optim

from cns.models.graph_vs import GraphVS
from cns.ablation.temporal_transformer.graph_vs_transformer import GraphVS_Transformer
from cns.train_gvs_short_seq import train as train_short
from cns.train_gvs_long_seq import train as train_long


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GraphVS with Transformer temporal memory"
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64 for short, 8 for long)")
    parser.add_argument("--epochs", type=int, default=160,
                        help="Training epochs")
    parser.add_argument("--init-lr", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to checkpoint for resuming")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save checkpoints")
    parser.add_argument("--long", action="store_true", default=False,
                        help="Use long sequence training")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable PyBullet GUI")

    # [新增] Transformer 特有参数
    parser.add_argument("--num-heads", type=int, default=4,
                        help="[Transformer] Number of MHA heads")
    parser.add_argument("--max-len", type=int, default=8,
                        help="[Transformer] Sliding window memory length")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="[Transformer] Attention + FFN dropout")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  GraphVS_Transformer Training")
    print("=" * 60)
    print(f"  num_heads : {args.num_heads}")
    print(f"  max_len   : {args.max_len}")
    print(f"  dropout   : {args.dropout}")
    print(f"  hidden_dim: 128")
    print(f"  sequence  : {'long' if args.long else 'short'}")
    print("=" * 60)

    # 选择短序列或长序列训练
    train_fn = train_long if args.long else train_short
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_fn(
        regress_norm=True,
        model_class=GraphVS_Transformer,  # <<< 唯一差异: 使用 Transformer 变体
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        hidden_dim=128,
        ckpt_path=args.load,
        save=args.save,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
