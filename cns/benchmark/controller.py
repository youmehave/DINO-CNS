import json
import sys
import torch
import torch.nn as nn
import numpy as np

# --- PyTorch & PyG Compatibility Patches for old checkpoints ---
if not hasattr(nn.Module, "_lazy_load_hook"):
    setattr(nn.Module, "_lazy_load_hook", None)

try:
    import torch_geometric.inspector
    sys.modules["torch_geometric.nn.conv.utils.inspector"] = torch_geometric.inspector
    original_implements = torch_geometric.inspector.Inspector.implements
    def patched_implements(self, func_name: str):
        if not hasattr(self, "_cls"):
            self._cls = None
        return original_implements(self, func_name)
    torch_geometric.inspector.Inspector.implements = patched_implements
except ImportError:
    pass

try:
    from torch_geometric.nn.conv import MessagePassing
    original_setstate = MessagePassing.__setstate__
    def patched_setstate(self, state):
        state.setdefault("decomposed_layers", 1)
        state.setdefault("_decomposed_layers", 1)
        state.setdefault("explain", False)
        original_setstate(self, state)
    MessagePassing.__setstate__ = patched_setstate
except ImportError:
    pass
# ---------------------------------------------------------------

from typing import Union, Dict
from ..models.graph_vs import GraphVS
from ..midend.graph_gen import GraphData
from ..ablation.ibvs.ibvs import IBVS


class GraphVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        self.device = torch.device(device)
        # self.net: GraphVS = torch.load(ckpt_path, map_location=self.device)["net"]
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # 为了避免 PyG 版本不兼容导致的一系列内置属性缺失报错（如 _user_args 等），
        # 我们强制初始化一个全新的 GraphVS 实例，然后仅仅把旧对象的权重（state_dict）加载进去。
        self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)

        if isinstance(ckpt, dict) and "net" in ckpt:
            if isinstance(ckpt["net"], torch.nn.Module):
                self.net.load_state_dict(ckpt["net"].state_dict())
            else:
                self.net.load_state_dict(ckpt["net"])
        elif hasattr(ckpt, "net") and isinstance(ckpt.net, torch.nn.Module):
            self.net.load_state_dict(ckpt.net.state_dict())
        elif isinstance(ckpt, torch.nn.Module):
            self.net.load_state_dict(ckpt.state_dict())
        else:
            self.net.load_state_dict(ckpt)
        
        self.net.eval()
        self.hidden = None

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)

            if getattr(data, "new_scene").any():
                print("[INFO] Got new scene, set hidden state to zero")
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            self.hidden = raw_pred[-1]
            vel = self.net.postprocess(raw_pred, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel


class IBVSController(object):
    def __init__(self, config_path: str):
        with open(config_path, "r") as fp:
            use_mean = json.load(fp)["use_mean"]
        self.ibvs = IBVS(use_mean)

    def __call__(self, data: GraphData) -> np.ndarray:
        return self.ibvs(data)


class ImageVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        from ..ablation.ibvs.raft_ibvs import RaftIBVS
        from ..reimpl import ICRA2018, ICRA2021
        
        self.device = torch.device(device)
        self.net: Union[ICRA2018, ICRA2021, RaftIBVS] = \
            torch.load(ckpt_path, map_location=self.device, weights_only=False)["net"]
        self.net.eval()
        self.tar_feat = None
    
    def __call__(self, data: Dict) -> np.ndarray:
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            
            if data.get("new_scene", True):
                self.tar_feat = None
            
            data["tar_feat"] = self.tar_feat
            raw_pred = self.net(data)
            self.tar_feat = data["tar_feat"]
            vel = self.net.postprocess(raw_pred, data)
        
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        vel = vel.flatten()
        return vel
