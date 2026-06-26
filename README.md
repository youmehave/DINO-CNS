# CNS + DINO: Correspondence Encoded Neural Image Servo with DINO Frontend

> **Acknowledgement**  
> This project is built upon the excellent work of **[CNS: Correspondence Encoded Neural Image Servo Policy](https://github.com/hhcaz/CNS)** (Chen et al., 2023). All original CNS model architecture, training pipeline, and evaluation benchmarks remain intact. Our contributions focus on integrating **[DINO/DINOv2](https://github.com/facebookresearch/dino)** (Caron et al., 2021) as a frontend detector and extending the temporal modeling with Transformer-based memory — see `cns/ablation/` for all additions.
>
> 🙏 Thanks to the CNS authors for open-sourcing their work, and to the DINO team at Facebook Research for the self-supervised vision transformer models.

---

## Introduction

*This is the original CNS introduction, preserved verbatim from the upstream repository.*

This is the official implementation of our paper "CNS: Correspondence Encoded Neural Image Servo Policy". We present a graph neural network based solution for image servo utilizing explicit keypoints correspondence obtained from any detector-based feature matching methods, such as SIFT, AKAZE, ORB, SuperGlue and etc. 

<p align="center">
<img src="README.assets/cns_pipeline.png" width="500">
</p>

Our model achieves <0.3° and sub-millimeter precision in real-world experiments (mean distance to target ≈ 0.275m) and runs in real-time (~40 fps with ORB as front-end).

* Full paper: https://arxiv.org/abs/2309.09047
* Homepage: https://hhcaz.github.io/CNS-home
* Video: https://www.bilibili.com/video/BV1cK4y1F7un

We provide the pre-trained model in `checkpoints` folder. See demo in script `demo_sim_Erender.py` which launches 150 servo episodes in simulation environment ${\rm E}_{\rm render}$ as described in the main text. We use `demo_real_ur5.py` to benchmark methods in real-world environments but you may need to adapt the code to fit your own robot.



## Dependencies

* **(Required)** We use [PyTorch](https://pytorch.org) (>1.12) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (PyTorch Geometric). Please follow their official guidelines to install them. Note the version of PyG should be compatible with PyTorch. Here are the additional dependencies, they can be installed via pip:

  ```
  pip install tqdm numpy scipy pybullet matplotlib tensorboard scikit-image open3d>=0.12.0 opencv-python>=4.8.0 pyrealsense2==2.53.1.4623
  ```

* (Optional, YCB objects)  If you want to use the same simulation environment setup ${\rm E}_{\rm render}$ as this work, you need to download the YCB object models from [this repo](https://github.com/eleramp/pybullet-object-models) and put them in folder `cns/thirdparty/`:

  ```
  cd cns/thirdparty
  git clone https://github.com/eleramp/pybullet-object-models.git
  ```

* (Optional, SuperGlue) If you want to use SuperGlue as the observer, you need to manually download it from [this repo](https://github.com/magicleap/SuperGluePretrainedNetwork.git):

  ```
  cd cns/thirdparty
  git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
  python prepare_superglue.py
  ```
  
**Note:** the original implementation of SuperGlue may fail with large in-plane rotation. We follow the solution in [this issue](https://github.com/magicleap/SuperGluePretrainedNetwork/issues/59) to rotate the desired image multiple times for matching and then rotate the matched keypoints back. Thanks to [kyuhyoung](https://github.com/kyuhyoung/SuperGluePretrainedNetwork) for providing scripts. 
  
When initializing the front-end, you need to specify `detector="SuperGlue:0123"` to enable this feature. (`SuperGlue:0123` uses images rotated 3 times, `SuperGlue:02` uses the original image and image rotated by 180°, `SuperGlue` uses original image only.)

* (Optional, DINO/DINOv2) If you want to use DINO or DINOv2 as the frontend observer:

  ```
  pip install timm
  ```
  
  The DINO model weights are downloaded automatically on first use. Available config strings: `dino_vits16:16` (fast), `dino_vits8:8` (high-res), `dino_vitb8:8` (best quality).



## How to Use

There are three steps to go to use CNS in a general image servo task.

### 1. Initialize the pipeline

First initialize the pipeline, this could be direct instantiation with proper arguments:

```python
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt

pipeline = CorrespondenceBasedPipeline(
    ckpt_path="<path-to-the-checkpoint>",
    detector="SIFT",  # this could be ORB, AKAZE, SuperGlue, or dino_vits16:16
    device="cuda:0",  # or "cpu" if cuda is not available
    intrinsic=CameraIntrinsic.default(),  # we use default camera intrinsic here, 
                                          # changes to your camera intrinsic
    ransac=True,  # whether conduct ransac after keypoints detection and matching
    vis=VisOpt.ALL  # can be KP (keypoints), MATCH, GRAPH and their combinitions, or NO
)
```

or loading from a `json` file:

```python
pipeline = CorrespondenceBasedPipeline.from_file("pipeline.json")
```

The `json` file looks like:

```json
{
    "intrinsic": {
        "width": 640,
        "height": 480,
        "K": [
            615.7113,      0.0, 315.7990,  // fx  0 cx
                 0.0, 615.7556, 248.1492,  //  0 fy cy
                 0.0,      0.0,      1.0   //  0  0  1
        ]
    },
    "detector": "SIFT",
    "checkpoint": "checkpoints/cns.pth",
    "device": "cuda:0",
    "visualize": "MATCH|GRAPH"
}
```

**Note:** Currently enabling the visualization will introduce extra time delay in the control system, the controller may damp around the desired pose and takes longer time to convergence (to stop itself).

### 2. Specify the desired image and distance prior

The desired image is a numpy array of `shape = (H, W, 3)` and `dtype = uint8`. Channels are ordered in BGR format. The distance prior is a scalar representing the distance from camera to scene center in the desired pose. The distance prior can be roughly estimated if the ground truth value is hard to obtain (we recommend to **underestimate** the value if your are not sure about the ground truth, for example, if the ground truth is 0.5m, you can set this to a value between 0.25~0.5).

```python
pipeline.set_target(
    desired_image,  # numpy array, shape = (H, W, 3), dtype = uint8, BGR format
    distance_prior  # a scalar
)
```

### 3. Get current observation, calculate the velocity control rate and conduct it

User needs to implement the details of `get_obervation` and `conduct_velocity_control`. Method `pipeline.get_control_rate` returns: (1) a 6-DoF camera velocity in camera frame $[v_x, v_y, v_z, w_x, w_y, w_z]$, (2) data representing the graph structure and (3) a dictionary recording time cost on front-end, graph construction and neural network forward.

```python
while True:
    current_image = get_observation()
    velocity, data, timing = pipeline.get_control_rate(current_image)
    conduct_velocity_control(velocity)
```

That's all. But you may want to know when is appropriate to terminate the servo episode. We provide two stop policy: `PixelStopPolicy` and `SSIMStopPolicy`. The first one evaluates the points position error of detected keypoints between current image and desired image, while the second one evaluates the SSIM between current and desired image. Adding stop policy to the servo process yields:

```python
import time
from cns.benchmark.stop_policy import PixelStopPolicy, SSIMStopPolicy

stop_policy = PixelStopPolicy(
    waiting_time=2.0,  # 2 secs
    conduct_thresh=0.01  # error threshold
)
# # or use SSIMStopPolicy policy
# stop_policy = SSIMStopPolicy(
#     waiting_time=2.0,  # 2 secs
#     conduct_thresh=0.1  # error threshold
# )

stop_policy.reset()  # need to be called right before starting the servo process
while True:
    current_image = get_observation()
    velocity, data, timing = pipeline.get_control_rate(current_image)
    if stop_policy(data, time.time()):
        break
    conduct_velocity_control(velocity)
```

The `stop_policy` calculate the error and if the error is lower than `conduct_thresh` and doesn't decrease anymore for certain time (specified by `waiting_time`), it returns `True` to indicate that it's time to terminate. We don't use fixed error threshold for termination because different front-ends give different qualities of keypoints and correspondence. We'd rather use an adaptive scheme for termination when the error is relative small and doesn't decrease (is larger than the historical minimum value) for a while. You can also use your own stop policy with necessary information stored in `data` (including current and desired point positions and images).

**Note:** `PixelStopPolicy` calculate L2 norm of keypoints position error (in **normalized image plane**).



## Training and Evaluation

**1. Training**

First train CNS with short trajectory sequences for faster convergence (~9h on single RTX 2080Ti):

```shell
python train_cns.py --batch-size=64 --epochs=50 --init-lr=1e-3 --weight-decay=1e-4 --device="cuda:0" --gui --save
```

The trained model is saved as, for example, `checkpoints/datetime_CNS/checkpoint_best.pth`. The model can already be used for servoing. But train it with longer sequences will improve the final precision (costs another ~6h):

```shell
python train_cns.py --batch-size=16 --epochs=50 --init-lr=1e-4 --weight-decay=1e-4 --device="cuda:0" --load=checkpoints/datetime_CNS/checkpoint_best.pth --gui --long --save
```

You can see the loss curve via:

```
tensorboard --logdir="logs"
```

![loss_curve](README.assets/loss_curve.png)

**2. Evaluation**

Please follow the example script in `cns/benchmark/tests.py` to prepare checkpoints and environment.



---

## Extended: Our Improvements on CNS

> The following sections document the extensions we have made on top of the original CNS:
> - **DINO/DINOv2 frontend** for dense semantic feature matching
> - **GraphVS_Transformer** with MultiheadAttention temporal memory
> - **DINO fine-tuning pipeline** for domain adaptation
>
> All additions are in `cns/ablation/` and `cns/frontend/dino_*` — the original CNS codebase remains unmodified.

### Environment Setup Summary

```bash
# Create conda environment
conda create -n cns python=3.10 -y && conda activate cns

# PyTorch (choose based on your GPU)
# RTX 2080Ti / 3090 (CUDA 11.x):
pip install torch torchvision torchaudio
# RTX 4090 (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric + dependencies
pip install torch_geometric
pip install tqdm numpy scipy pybullet matplotlib tensorboard scikit-image open3d>=0.12.0 opencv-python>=4.8.0 timm

# Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Supported Frontend Detectors

| `detector` value | Backend | Notes |
|---|---|---|
| `SIFT` | OpenCV SIFT | Default, fast |
| `AKAZE` | OpenCV AKAZE | Fast |
| `ORB` | OpenCV ORB | Very fast |
| `SuperGlue:0123` | SuperGlue + SuperPoint | 4 rotation aug |
| `dino_vits16:16` | DINO ViT-S/16 | **New!** |
| `dino_vitb8:8` | DINOv2 ViT-B/8 | **New!** |
| `SIFT+ORB` | Concatenated | Combine multiple |

### Train GraphVS_Transformer (New Architecture)

The Transformer variant replaces the GRU temporal module with a MultiheadAttention-based sliding-window memory. The training/inference interface is identical to the original GraphVS.

```bash
# Short sequence (~4h on RTX 4090)
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 64 --epochs 160 --save --num-heads 4 --max-len 8

# Long sequence (~3h on RTX 4090)
python -m cns.ablation.temporal_transformer.train_graph_vs_transformer \
    --batch-size 8 --epochs 80 --long \
    --load checkpoints/<short_best>.pth --save
```

| Hyperparameter | Default | Description |
|---|---|---|
| `--num-heads` | 4 | MHA attention heads |
| `--max-len` | 8 | Sliding window memory frames |
| `--dropout` | 0.0 | Attention + FFN dropout |

> **Note:** Architecture differs from original GraphVS — **must train from scratch**, cannot load `cns.pth` weights.

### Model Variants

All subclasses `GraphVS` and maintain the same external interface. See `cns/ablation/`:

| Model | Location | Difference from Original |
|---|---|---|
| `GraphVS` | `cns/models/graph_vs.py` | Original (GRU-based) |
| `GraphVS_Transformer` | `cns/ablation/temporal_transformer/` | **New:** Transformer temporal memory |
| `GraphVS_EdgeConv` | `cns/ablation/structure/` | EdgeConv replaces PERConv |
| `GraphVS_SimpleGRU` | `cns/ablation/structure/` | Simpler GRU |
| `GraphVS_woGRU` | `cns/ablation/structure/` | No temporal module |
| `GraphVS_NoCluster` | `cns/ablation/cluster/` | No point clustering |

### DINO Frontend Fine-tuning

The original model was trained with synthetic noise (simulating SIFT dropout/mismatch). When deploying with DINO frontend, fine-tuning on real DINO correspondences from simulation bridges the domain gap.

```bash
# Fine-tune original GraphVS with DINO data (~2h on RTX 4090)
python -m cns.ablation.dino_adapt.dino_finetune \
    --ckpt checkpoints/cns.pth --epochs 20 --save

# Fine-tune Transformer variant with DINO data
python -m cns.ablation.dino_adapt.dino_finetune \
    --ckpt checkpoints/transformer_best.pth --epochs 20 --save \
    --output checkpoints/dino_transformer_finetune.pth
```

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `checkpoints/cns.pth` | Pre-trained checkpoint |
| `--dino-config` | `dino_vits16:16` | DINO model config |
| `--epochs` | 20 | Fine-tuning epochs |
| `--steps-per-epoch` | 100 | Steps per epoch |
| `--lr` | 1e-4 | Learning rate |
| `--teacher-ratio` | 0.3 | Teacher forcing probability |
| `--save` | (flag) | Save best checkpoint |
| `--output` | auto | Custom output path |

### RTX 4090 Training Workflow

Total time: **~9 hours** on a single RTX 4090.

```
Stage 1 (4h):   GraphVS_Transformer short sequence training
Stage 2 (3h):   GraphVS_Transformer long sequence training
Stage 3 (2h):   DINO frontend fine-tuning
```

### Project Structure (New Additions)

```
cns/ablation/
├── temporal_transformer/    # [NEW] Transformer temporal memory variant
│   ├── graph_vs_transformer.py
│   └── train_graph_vs_transformer.py
├── dino_adapt/              # [NEW] DINO fine-tuning pipeline
│   └── dino_finetune.py
cns/frontend/
│   ├── dino_frontend.py     # [NEW] DINO/DINOv2 dense feature matching
│   └── dinov2_extractor.py  # [NEW] ViT feature extractor
```



## Acknowledgement

We use the following repositories in this project:

* [pybullet-object-models](https://github.com/eleramp/pybullet-object-models): We use YCB object models to create servo scenes in simulated environment.
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork): We use SuperGlue as a candidate observer.
* [DINO](https://github.com/facebookresearch/dino): Self-supervised Vision Transformers for dense feature matching.



## BibTex Citation

If you found it helpful to you, please consider citing the original CNS paper and DINO:

```
@misc{chen2023cns,
      title={CNS: Correspondence Encoded Neural Image Servo Policy}, 
      author={Anzhe Chen and Hongxiang Yu and Yue Wang and Rong Xiong},
      year={2023},
      eprint={2309.09047},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

@inproceedings{caron2021dino,
      title={Emerging Properties in Self-Supervised Vision Transformers},
      author={Mathilde Caron and Hugo Touvron and Ishan Misra and Herv\'e J\'egou 
              and Julien Mairal and Piotr Bojanowski and Armand Joulin},
      booktitle={ICCV},
      year={2021}
}
```
