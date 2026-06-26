# CNS Training and Inference Demos

## Introduction

This is the official implementation of our paper **"CNS: Correspondence Encoded Neural Image Servo Policy"**. We present a graph neural network based solution for image servo utilizing explicit keypoints correspondence obtained from any detector-based feature matching methods, such as SIFT, AKAZE, ORB, SuperGlue, and DINO/DINOv2.

<p align="center">
<img src="README.assets/cns_pipeline.png" width="500">
</p>

Our model achieves <0.3° and sub-millimeter precision in real-world experiments (mean distance to target ≈ 0.275m) and runs in real-time (~40 fps with ORB as front-end).

- Full paper: https://arxiv.org/abs/2309.09047
- Homepage: https://hhcaz.github.io/CNS-home
- Video: https://www.bilibili.com/video/BV1cK4y1F7un

---

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n cns python=3.10 -y
conda activate cns
```

### 2. Install PyTorch

| GPU | Command |
|---|---|
| RTX 2080Ti / 3090 (CUDA 11.x) | `pip install torch torchvision torchaudio` |
| RTX 4090 (CUDA 12.1) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric
```

### 4. Install Other Dependencies

```bash
pip install tqdm numpy scipy pybullet matplotlib tensorboard scikit-image open3d>=0.12.0 opencv-python>=4.8.0
```

(Optional) RealSense camera support:

```bash
pip install pyrealsense2==2.53.1.4623
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import torch_geometric; print('PyG OK')"
```

### 6. (Optional) YCB Object Models

For simulation with rendered objects:

```bash
cd cns/thirdparty
git clone https://github.com/eleramp/pybullet-object-models.git
```

### 7. (Optional) SuperGlue Frontend

```bash
cd cns/thirdparty
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
python prepare_superglue.py
```

### 8. (Optional) DINO/DINOv2 Frontend

The DINO frontend uses the `timm` library and downloads model weights automatically on first use:

```bash
pip install timm
```

Model options:
| Config String | Model | Stride | Notes |
|---|---|---|---|
| `dino_vits16:16` | DINO ViT-S/16 | 16 | Default, fast |
| `dino_vits8:8` | DINO ViT-S/8 | 8 | Higher resolution |
| `dino_vitb8:8` | DINO ViT-B/8 | 8 | Best quality, slower |

---

## Quick Start: Inference

### Using the Pipeline

```python
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt

# Load pipeline from config file
pipeline = CorrespondenceBasedPipeline.from_file("pipeline.json")

# Set target image
pipeline.set_target(desired_image, distance_prior=0.5)

# Servo loop
while True:
    velocity, data, timing = pipeline.get_control_rate(current_image)
    # conduct_velocity_control(velocity)
```

### Pipeline Configuration (`pipeline.json`)

```json
{
    "intrinsic": {
        "width": 640,
        "height": 480,
        "K": [615.7113, 0.0, 315.7990,
              0.0, 615.7556, 248.1492,
              0.0, 0.0, 1.0]
    },
    "detector": "dino_vits16:16",
    "checkpoint": "checkpoints/cns.pth",
    "device": "cuda:0",
    "visualize": "MATCH"
}
```

### Supported Frontends

| `detector` value | Backend | Notes |
|---|---|---|
| `SIFT` | OpenCV SIFT | Default, fast |
| `AKAZE` | OpenCV AKAZE | Fast |
| `ORB` | OpenCV ORB | Very fast |
| `SuperGlue:0123` | SuperGlue + SuperPoint | 4 rotation aug |
| `dino_vits16:16` | DINO ViT-S/16 | **New!** Semantic matching |
| `dino_vitb8:8` | DINOv2 ViT-B/8 | **New!** Best quality |
| `SIFT+ORB` | Concatenated | Combine multiple |

### Stop Policy

```python
from cns.benchmark.stop_policy import PixelStopPolicy

stop_policy = PixelStopPolicy(waiting_time=2.0, conduct_thresh=0.01)
stop_policy.reset()

while True:
    velocity, data, timing = pipeline.get_control_rate(current_image)
    if stop_policy(data, time.time()):
        break
    conduct_velocity_control(velocity)
```

---

## Training

### 1. Train Original GraphVS (GRU-based)

**Short sequence training** (~9h on RTX 2080Ti, ~4h on RTX 4090):

```bash
python train_cns.py \
    --batch-size 64 \
    --epochs 160 \
    --init-lr 5e-4 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --gui --save
```

**Long sequence training** (continue from short, ~6h / ~3h on 4090):

```bash
python train_cns.py \
    --batch-size 8 \
    --epochs 80 \
    --init-lr 1e-4 \
    --device cuda:0 \
    --load checkpoints/<short_seq_best>.pth \
    --long --gui --save
```

Monitor training:

```bash
tensorboard --logdir logs/
```

![loss_curve](README.assets/loss_curve.png)

### 2. Train GraphVS_Transformer (NEW)

The Transformer variant replaces the GRU temporal module with a MultiheadAttention-based memory mechanism. All training/inference interfaces are identical to the original.

**Short sequence training:**

```bash
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 64 \
    --epochs 160 \
    --save \
    --num-heads 4 \
    --max-len 8
```

**Long sequence training:**

```bash
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 8 \
    --epochs 80 \
    --long \
    --load checkpoints/<transformer_short_best>.pth \
    --save
```

**Transformer hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `--num-heads` | 4 | Multi-head attention heads |
| `--max-len` | 8 | Sliding window memory length |
| `--dropout` | 0.0 | Attention + FFN dropout rate |

> **Note:** The Transformer model must be trained **from scratch** — it cannot load original GraphVS weights due to architecture differences.

### 3. Model Variants (Ablation Studies)

All variants follow the same training interface. See `cns/ablation/` for full list:

| Model | Location | Description |
|---|---|---|
| `GraphVS` | `cns/models/graph_vs.py` | Original (GRU-based) |
| `GraphVS_Transformer` | `cns/ablation/temporal_transformer/` | **NEW** Transformer memory |
| `GraphVS_EdgeConv` | `cns/ablation/structure/` | EdgeConv replaces PERConv |
| `GraphVS_SimpleGRU` | `cns/ablation/structure/` | Simpler GRU variant |
| `GraphVS_woGRU` | `cns/ablation/structure/` | No temporal module |
| `GraphVS_NoCluster` | `cns/ablation/cluster/` | No point clustering |

---

## DINO Frontend Fine-tuning

### Why Fine-tune?

The original model was trained with synthetic noise patterns (simulating SIFT-like keypoint dropout/mismatch). When using the DINO frontend at deployment, there is a domain gap:

| | Training (synthetic) | Deployment (DINO) |
|---|---|---|
| Error type | Random dropout + Gaussian noise | Systematic bias in textureless regions |
| Correspondence density | ~50-300 random points | ~100-300 SIFT points + DINO matching |
| Temporal consistency | Independent per frame | Smooth but with persistent bias |

Fine-tuning bridges this gap by training on actual DINO correspondences from rendered simulation images.

### DINO Fine-tuning Command

```bash
python -m cns.ablation.dino_adapt.dino_finetune \
    --ckpt checkpoints/cns.pth \
    --epochs 20 \
    --save
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `checkpoints/cns.pth` | Pre-trained checkpoint to fine-tune |
| `--dino-config` | `dino_vits16:16` | DINO model config |
| `--epochs` | 20 | Fine-tuning epochs (~2h total) |
| `--steps-per-epoch` | 100 | Training steps per epoch |
| `--lr` | 1e-4 | Learning rate (lower than scratch training) |
| `--teacher-ratio` | 0.3 | Probability of using ground-truth velocity |
| `--save` | (flag) | Save checkpoint |
| `--output` | `checkpoints/dino_finetune_best.pth` | Output path |
| `--gui` | (flag) | Show PyBullet debug GUI |

### Fine-tune the Transformer Variant with DINO

```bash
python -m cns.ablation.dino_adapt.dino_finetune \
    --ckpt checkpoints/transformer_best.pth \
    --epochs 20 \
    --save \
    --output checkpoints/dino_transformer_finetune.pth
```

### Using the Fine-tuned Model

```json
{
    "checkpoint": "checkpoints/dino_finetune_best.pth",
    "detector": "dino_vits16:16",
    ...
}
```

---

## RTX 4090 Deployment Guide

### Transfer Code to 4090

```bash
# On the 4090 machine:
git clone https://github.com/<your-username>/DINO-CNS.git
cd DINO-CNS
git checkout dev/dino-ros-integration
```

### Environment Setup on 4090

```bash
conda create -n cns python=3.10 -y
conda activate cns

# PyTorch for CUDA 12.1 (4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric

# Other dependencies
pip install tqdm numpy scipy pybullet matplotlib tensorboard scikit-image \
    open3d>=0.12.0 opencv-python>=4.8.0 timm

# Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 4090
```

### Training Workflow on 4090

**Stage 1** — Train GraphVS_Transformer from scratch (~4h):

```bash
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 64 --epochs 160 --save
```

**Stage 2** — Long sequence training (~3h):

```bash
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 8 --epochs 80 --long \
    --load checkpoints/<stage1_best>.pth --save
```

**Stage 3** — DINO fine-tuning (~2h):

```bash
pip install timm  # if not already installed

python -m cns.ablation.dino_adapt.dino_finetune \
    --ckpt checkpoints/<stage2_best>.pth \
    --epochs 20 --save
```

**Total time: ~9 hours on RTX 4090.**

### Transfer Results Back

```bash
# On 4090:
scp checkpoints/dino_transformer_finetune.pth user@your-machine:/path/to/DINO_CNS/checkpoints/
```

### 4090-Specific Optimization

The 4090 has 24GB VRAM — you can use larger batch sizes:

```bash
# Short sequence: increase batch size
--batch-size 128  # (was 64 for 2080Ti)

# Long sequence: increase batch size
--batch-size 16   # (was 8 for 2080Ti)

# Transformer: more memory for longer temporal context
--max-len 16      # (was 8 for 2080Ti)
```

---

## Project Structure

```
DINO_CNS/
├── cns/
│   ├── frontend/          # Feature detectors (SIFT, SuperGlue, DINO)
│   │   ├── classic.py     # OpenCV-based (SIFT/ORB/AKAZE)
│   │   ├── superglue.py   # SuperGlue + SuperPoint
│   │   └── dino_frontend.py  # [NEW] DINO/DINOv2 dense matching
│   ├── midend/            # Correspondence → Graph conversion
│   ├── models/            # GNN architecture (GraphVS)
│   │   └── graph_vs.py    # Original model (UNMODIFIED)
│   ├── sim/               # Simulation training environments
│   ├── benchmark/         # Evaluation pipeline & controllers
│   ├── real/              # Real UR5 robot interface
│   ├── ablation/          # Model variants & experiments
│   │   ├── structure/     # Backbone variants
│   │   ├── cluster/       # Graph structure variants
│   │   ├── temporal_transformer/  # [NEW] Transformer memory
│   │   └── dino_adapt/    # [NEW] DINO fine-tuning pipeline
│   └── utils/             # Camera, trainer, visualization
├── checkpoints/           # Pre-trained model weights
├── pipeline.json          # Default pipeline config
└── train_cns.py           # Original training entry point
```

---

## Acknowledgement

We use the following repositories in this project:

- [pybullet-object-models](https://github.com/eleramp/pybullet-object-models): YCB object models for simulation.
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork): SuperGlue feature matching.
- [DINO](https://github.com/facebookresearch/dino): Self-supervised Vision Transformers.

---

## BibTex Citation

If you found it helpful to you, please consider citing:

```
@misc{chen2023cns,
      title={CNS: Correspondence Encoded Neural Image Servo Policy},
      author={Anzhe Chen and Hongxiang Yu and Yue Wang and Rong Xiong},
      year={2023},
      eprint={2309.09047},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
