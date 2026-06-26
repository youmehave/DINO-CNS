#!/usr/bin/env python3
"""
================================================================================
  dino_finetune.py — 用 DINO 前端数据微调 CNS 模型
================================================================================

  【目的】
  原始模型使用合成噪声（模拟 SIFT 的随机丢失/误匹配）训练，部署时切换到 DINO
  前端会产生域差异（DINO 在弱纹理区的系统性偏移、不同的关键点分布等）。
  本脚本在仿真环境中用真实 DINO 前端提取对应关系，对预训练模型进行微调。

  【原理】
  仿真渲染图像 → DINO 前端提取对应关系 → Midend 图构建 → GraphVS 推理
      ↑                                                         │
      └────────── 仿真器（supervisor 提供真值速度）────────────────┘

  【零侵入】
  - 不修改 cns/models/graph_vs.py
  - 不修改 cns/sim/dataset.py
  - 仅新增 DINO 数据生成 + 微调逻辑

  【用法】
  # 短序列微调（推荐先用这个，~2h）
  python -m cns.ablation.dino_adapt.dino_finetune \
      --ckpt checkpoints/cns.pth \
      --epochs 20 \
      --save

  # 长序列微调（短序列之后继续，~1h）
  python -m cns.ablation.dino_adapt.dino_finetune \
      --ckpt checkpoints/xxx_dino_short_best.pth \
      --epochs 10 --long --save
================================================================================
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pybullet as p
from torch_geometric.data import Batch

# 项目内部模块
from cns.models.graph_vs import GraphVS
from cns.midend.corr2graph import Midend
from cns.midend.graph_gen import GraphData
from cns.frontend.dino_frontend import DINOFrontend
from cns.utils.perception import CameraIntrinsic, Camera
from cns.sim.environment import ImageEnvGUI
from cns.sim.supervisor import supervisor_vel, Policy


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DINO 数据生成器                                                            ║
# ║  仿真渲染图像 → DINO 前端 → Correspondence → GraphData                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DINODataGenerator(object):
    """
    使用 PyBullet 仿真环境渲染图像，通过 DINO 前端提取对应关系，
    生成带真值速度标签的 GraphData 训练数据。
    """

    def __init__(self, dino_config="dino_vits16:16", intrinsic=None,
                 gui=False, resample=True):
        """
        参数:
            dino_config: DINO 前端配置字符串，如 "dino_vits16:16"
            intrinsic:   CameraIntrinsic 或 None（使用默认值 640x480）
            gui:         是否显示 PyBullet GUI（调试用）
            resample:    场景切换时是否重新采样物体
        """
        self.gui = gui

        # 初始化仿真环境（渲染 YCB 物体）
        client = p.SHARED_MEMORY if gui else p.DIRECT
        self.client = p.connect(client)
        if not gui:
            p.setRealTimeSimulation(0, physicsClientId=self.client)

        # 相机内参
        if intrinsic is None:
            intrinsic = CameraIntrinsic.default()
        self.intrinsic = intrinsic

        # DINO 前端
        print(f"[INFO] Initializing DINO frontend: {dino_config}")
        self.frontend = DINOFrontend(intrinsic, dino_config, ransac=True)

        # 图构建器
        self.midend = Midend()

        # 仿真环境（复用 ImageEnvGUI 的场景生成和相机采样）
        self.env = ImageEnvGUI(intrinsic, resample=resample, auto_reinit=False)

        # 状态
        self.tar_img = None
        self.current_wcT = None
        self.target_wcT = None

    def init_scene(self):
        """初始化新场景：摆放 YCB 物体，采样相机姿态，渲染目标图像"""
        self.env.init()
        self._capture_target()
        return True

    def _capture_target(self):
        """从仿真环境渲染目标图像并设置到 DINO 前端"""
        target_cwT = np.linalg.inv(self.env.target_wcT)
        frame = self.env.camera.render(target_cwT, self.client)
        rgb = frame.color_image()
        self.tar_img = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR
        self.frontend.update_target_frame(self.tar_img)
        self.target_wcT = self.env.target_wcT.copy()

    def step(self):
        """
        执行一步仿真 → 返回训练数据

        返回:
            data:  GraphData（包含图结构 + 真值速度）
            或 None（DINO 前端匹配失败时）
        """
        env = self.env
        current_cwT = np.linalg.inv(env.current_wcT)

        # 1. 渲染当前帧
        frame = env.camera.render(current_cwT, env.client)
        rgb = frame.color_image()
        cur_img = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR

        # 2. DINO 前端提取对应关系
        corr = self.frontend.process_current_frame(cur_img)
        if corr is None:
            print("[WARN] DINO frontend returned None (no matches)")
            return None

        # 3. Midend 构建图
        data = self.midend.get_graph_data(corr)
        if data is None:
            print("[WARN] Midend returned None")
            return None

        # 4. 获取真值速度（使用目标帧的关键点对应的 3D 点）
        target_cwT = np.linalg.inv(env.target_wcT)

        # 提取有效的匹配点到归一化坐标
        tar_pos = corr.tar_pos           # (N, 2) 像素坐标
        cur_pos_aligned = corr.cur_pos_aligned  # (N, 2) 像素坐标（已对齐到 tar 索引）
        valid_mask = corr.valid_mask     # (N,) bool

        # 获取 3D 点（从仿真环境的 depth 图中采样）
        W, H = self.intrinsic.width, self.intrinsic.height
        pc = np.asarray(frame.point_cloud().points).reshape(H, W, 3)

        # 为每个有效关键点采样 3D 坐标
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            print("[WARN] No valid correspondences")
            return None

        # 对有效关键点采样深度
        valid_tar_ji = np.clip(
            np.round(tar_pos[valid_indices]).astype(np.int32),
            [0, 0], [W - 1, H - 1]
        )
        cc, rr = valid_tar_ji.T
        points_3d = np.ascontiguousarray(pc[rr, cc])  # (M, 3)

        # 只保留深度有效的点
        depth_valid = (points_3d[:, -1] > 0.01) & (points_3d[:, -1] < 10.0)
        if depth_valid.sum() < 4:
            print("[WARN] Too few valid depth points")
            return None

        # 5. 计算真值速度（PBVS supervisor）
        # 获取所有点的像素坐标
        current_xy = cur_pos_aligned[valid_indices][depth_valid]  # (K, 2)
        target_xy = tar_pos[valid_indices][depth_valid]           # (K, 2)
        pbvs_points = points_3d[depth_valid]                      # (K, 3)

        # 计算 Z 值
        current_Z = (pbvs_points @ current_cwT[:3, :3].T + current_cwT[:3, 3])[:, -1]
        target_Z = (pbvs_points @ target_cwT[:3, :3].T + target_cwT[:3, 3])[:, -1]

        vel, (tPo_norm, vel_si) = supervisor_vel(
            Policy.PBVS_Straight,
            current_xy, current_Z, target_xy, target_Z,
            self.intrinsic, env.current_wcT, env.target_wcT, pbvs_points
        )

        # 6. 附加训练标签到 GraphData
        data.start_new_scene()
        data.set_distance_scale(tPo_norm)
        setattr(data, "vel", torch.from_numpy(vel[None, :]).float())
        setattr(data, "vel_si", torch.from_numpy(vel_si[None, :]).float())
        setattr(data, "tPo_norm", torch.tensor([tPo_norm]).float())
        setattr(data, "new_scene", torch.tensor([False]))

        # 额外信息（调试用）
        setattr(data, "cur_img_np", cur_img)
        setattr(data, "tar_img_np", self.tar_img)
        setattr(data, "num_valid", torch.tensor([depth_valid.sum()]))

        return data

    def feedback(self, vel, norm=True):
        """将预测速度反馈给仿真环境以更新相机姿态"""
        if isinstance(vel, torch.Tensor):
            vel = vel.detach().cpu().numpy().squeeze()
        if norm:
            v_norm = np.linalg.norm(vel[:3]) + 1e-8
            w_norm = np.linalg.norm(vel[3:]) + 1e-8
            max_v = 0.5   # 最大线速度 0.5 m/s
            max_w = 0.5   # 最大角速度 0.5 rad/s
            if v_norm > max_v:
                vel[:3] = vel[:3] / v_norm * max_v
            if w_norm > max_w:
                vel[3:] = vel[3:] / w_norm * max_w
        self.env.action(vel)

    def need_reinit(self):
        """判断是否需要重新初始化场景"""
        return self.env.need_reinit()

    def close(self):
        """清理资源"""
        p.disconnect(self.client)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  微调训练流水线                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DINOFineTunePipeline(object):
    """
    DINO 数据微调流水线

    加载预训练 GraphVS 权重，使用 DINO 前端生成的仿真数据进行微调。
    保持与原始训练相同的损失函数和学习策略。
    """

    def __init__(self, ckpt_path, device, dino_config="dino_vits16:16",
                 gui=False, lr=1e-4):
        self.device = device

        # 加载预训练模型
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)

        # 兼容多种 checkpoint 格式
        if isinstance(ckpt, dict) and "net" in ckpt:
            if isinstance(ckpt["net"], torch.nn.Module):
                self.net.load_state_dict(ckpt["net"].state_dict())
            else:
                self.net.load_state_dict(ckpt["net"])
        elif hasattr(ckpt, "net"):
            if isinstance(ckpt.net, torch.nn.Module):
                self.net.load_state_dict(ckpt.net.state_dict())
            else:
                self.net.load_state_dict(ckpt.net)
        elif isinstance(ckpt, torch.nn.Module):
            self.net.load_state_dict(ckpt.state_dict())
        else:
            self.net.load_state_dict(ckpt)

        self.net.train()

        # 优化器（微调使用更低学习率）
        pg_wi_decay, pg_wo_decay = self.net.get_parameter_groups()
        self.optimizer = optim.AdamW([
            {"params": pg_wi_decay, "weight_decay": 1e-4},
            {"params": pg_wo_decay, "weight_decay": 0},
        ], lr=lr)

        # DINO 数据生成器
        self.generator = DINODataGenerator(
            dino_config=dino_config, gui=gui, resample=True
        )

        self.hidden = None
        self.epoch = 0

    def run_epoch(self, num_steps=100, teacher_ratio=0.3):
        """
        运行一个 epoch 的微调训练

        参数:
            num_steps:     每个 epoch 的训练步数
            teacher_ratio: Teacher forcing 比例（使用真值速度驱动仿真的概率）
        """
        losses = []
        results = []
        reinit_count = 0

        # 初始化场景
        self.generator.init_scene()
        self.hidden = None

        for step in range(num_steps):
            # 生成 DINO 训练数据
            try:
                data = self.generator.step()
            except Exception as e:
                print(f"[WARN] Step {step} data generation failed: {e}")
                traceback.print_exc()
                self.generator.env.action(np.zeros(6))
                continue

            if data is None:
                # DINO 匹配失败，使用零速度继续
                self.generator.env.action(np.zeros(6))
                self.generator.env.steps += 1
                continue

            data = data.to(self.device)

            # 跳过数据不足的样本
            if getattr(data, "num_valid", torch.tensor([0])).item() < 4:
                self.generator.env.action(np.zeros(6))
                self.generator.env.steps += 1
                continue

            # 前向传播
            if getattr(data, "new_scene").any() or (self.hidden is None):
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            self.hidden = raw_pred[-1]

            # 计算损失
            result, loss = self.net.objectives(raw_pred, data)
            losses.append(loss.item())
            for k in result:
                results.append({k: result[k]})

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止 DINO 噪声导致的梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.optimizer.step()

            # BPTT 截断（与原始训练一致）
            if (step + 1) % 8 == 0:
                if self.hidden is not None:
                    self.hidden = self.hidden.clone().detach()

            # 反馈速度驱动仿真
            pred_vel = self.net.postprocess(raw_pred, data)
            if np.random.random() < teacher_ratio:
                # Teacher forcing: 使用真值速度
                self.generator.feedback(getattr(data, "vel"), norm=True)
            else:
                self.generator.feedback(pred_vel, norm=True)

            # 检查是否需要重置场景
            if self.generator.need_reinit():
                reinit_count += 1
                self.generator.init_scene()
                self.hidden = None

            # 进度显示
            if (step + 1) % 20 == 0:
                avg_loss = sum(losses[-20:]) / len(losses[-20:])
                print(f"  Step {step+1:4d}/{num_steps}: "
                      f"loss={avg_loss:.4f}, "
                      f"mem={self.hidden.size(0) if self.hidden is not None else 0}, "
                      f"reinit={reinit_count}")

        avg_loss = sum(losses) / max(len(losses), 1)
        return avg_loss, reinit_count

    def save_checkpoint(self, path):
        """保存微调后的 checkpoint"""
        checkpoint = {
            "net": self.net,
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved to {path}")

    def close(self):
        self.generator.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  命令行入口                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CNS model with DINO frontend data"
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/cns.pth",
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--dino-config", type=str, default="dino_vits16:16",
                        help="DINO frontend config")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=100,
                        help="Training steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable PyBullet GUI (debug)")
    parser.add_argument("--teacher-ratio", type=float, default=0.3,
                        help="Teacher forcing ratio (0-1)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 创建微调流水线
    pipeline = DINOFineTunePipeline(
        ckpt_path=args.ckpt,
        device=device,
        dino_config=args.dino_config,
        gui=args.gui,
        lr=args.lr,
    )

    # 训练
    print("=" * 60)
    print("  DINO Fine-tuning Started")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps/epoch: {args.steps_per_epoch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Teacher ratio: {args.teacher_ratio}")
    print("=" * 60)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        pipeline.epoch = epoch
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        t0 = time.time()

        avg_loss, reinit_count = pipeline.run_epoch(
            num_steps=args.steps_per_epoch,
            teacher_ratio=args.teacher_ratio,
        )

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"time={elapsed:.0f}s, reinit={reinit_count}")

        # 保存最佳模型
        if args.save and avg_loss < best_loss:
            best_loss = avg_loss
            output_path = args.output or f"checkpoints/dino_finetune_best.pth"
            pipeline.save_checkpoint(output_path)

    # 保存最终模型
    if args.save:
        final_path = args.output or f"checkpoints/dino_finetune_epoch{args.epochs}.pth"
        pipeline.save_checkpoint(final_path)

    pipeline.close()
    print("\n[DONE] DINO fine-tuning completed.")


if __name__ == "__main__":
    main()
