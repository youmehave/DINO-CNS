import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .utils import FrontendBase, Correspondence
from .dinov2_extractor import ViTExtractor

class DINOFrontend(FrontendBase):
    def __init__(self, intrinsic, config="dino_vits16:16", ransac=True):
        self.intrinsic = intrinsic
        self.ransac = ransac
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_type = "dino_vits16"
        stride = 16
        if ":" in config:
            parts = config.split(":")
            model_type = parts[0]
            stride = int(parts[1])

        self.extractor = ViTExtractor(model_type=model_type, stride=stride, device=self.device)
        self.target_desc = None
        self.target_kp = None
        self.target_img = None
        self.patch_size = self.extractor.p
        self.stride = self.extractor.stride[0]
        self.tar_img_changed = True

    def get_upsampled_features(self, image):
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pad image to make it divisible by patch_size (required by DINOv2)
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            
        prep_img = self.extractor.preprocess_pil(img_rgb)
        
        with torch.no_grad():
            desc = self.extractor.extract_descriptors(prep_img.to(self.device), layer=11, facet='key')
            
        D = desc.shape[-1]
        padded_h, padded_w = img_rgb.shape[:2]
        num_patches_y = 1 + (padded_h - self.patch_size) // self.stride
        num_patches_x = 1 + (padded_w - self.patch_size) // self.stride
        
        feat_map = desc.view(num_patches_y, num_patches_x, D).permute(2, 0, 1).unsqueeze(0)
        
        # Upsample to the padded size, then crop back to original (h, w)
        feat_map_hr = F.interpolate(feat_map, size=(padded_h, padded_w), mode='bilinear', align_corners=False)
        feat_map_hr = feat_map_hr[:, :, :h, :w]
        
        feat_map_hr = F.normalize(feat_map_hr, p=2, dim=1)
        
        return feat_map_hr

    def update_target_frame(self, image, mask=None) -> bool:
        self.target_img = image
        self.tar_img_changed = True
        h, w = image.shape[:2]
        
        feat_map_hr = self.get_upsampled_features(image)
        
        sift = cv2.SIFT_create(nfeatures=300)
        kps = sift.detect(image, None)
        if len(kps) < 16:
            xs = np.linspace(16, w-16, 15)
            ys = np.linspace(16, h-16, 15)
            xv, yv = np.meshgrid(xs, ys)
            self.target_kp = np.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)
        else:
            self.target_kp = np.array([kp.pt for kp in kps], dtype=np.float32)
            
        grid = torch.tensor(self.target_kp, dtype=torch.float32, device=self.device)
        grid[:, 0] = (grid[:, 0] / (w - 1)) * 2 - 1.0
        grid[:, 1] = (grid[:, 1] / (h - 1)) * 2 - 1.0
        grid = grid.view(1, 1, -1, 2)
        
        sampled_desc = F.grid_sample(feat_map_hr, grid, align_corners=True)
        self.target_desc = sampled_desc.squeeze(0).squeeze(1).t()
        
        return True

    def process_current_frame(self, image) -> Correspondence:
        h, w = image.shape[:2]
        cur_feat_map_hr = self.get_upsampled_features(image)
        
        weights = self.target_desc.unsqueeze(-1).unsqueeze(-1)
        heatmap = F.conv2d(cur_feat_map_hr, weights).squeeze(0)
        
        heatmap_flat = heatmap.view(heatmap.shape[0], -1)
        max_idx = torch.argmax(heatmap_flat, dim=1)
        
        best_y = max_idx // w
        best_x = max_idx % w
        
        cur_kp = torch.stack([best_x, best_y], dim=1).float().cpu().numpy()
        
        matched_tar_kp = self.target_kp
        matched_cur_kp = cur_kp
        matched_tar_indices = np.arange(len(matched_tar_kp))
        matched_cur_indices = np.arange(len(matched_cur_kp))
        
        if self.ransac and len(matched_cur_kp) >= 4:
            E, mask = cv2.findEssentialMat(matched_cur_kp, matched_tar_kp, 
                                           self.intrinsic.K, method=cv2.RANSAC, 
                                           prob=0.999, threshold=2.0)
            if mask is not None:
                mask = mask.ravel().astype(bool)
                matched_tar_kp = matched_tar_kp[mask]
                matched_cur_kp = matched_cur_kp[mask]
                matched_tar_indices = matched_tar_indices[mask]
                matched_cur_indices = matched_cur_indices[mask]
                
        N_tar = self.target_kp.shape[0]
        valid_mask = np.zeros(N_tar, dtype=bool)
        cur_pos_aligned = np.zeros_like(self.target_kp)
        
        match = np.stack([matched_tar_indices, matched_cur_indices], axis=1) if len(matched_tar_indices) > 0 else np.zeros((0, 2), dtype=int)
        
        if len(matched_tar_indices) > 0:
            valid_mask[matched_tar_indices] = True
            cur_pos_aligned[matched_tar_indices] = matched_cur_kp

        tar_changed = self.tar_img_changed
        self.tar_img_changed = False

        return Correspondence(
            intrinsic=self.intrinsic,
            tar_img=self.target_img,
            tar_pos=self.target_kp,
            cur_img=image,
            cur_pos=cur_kp,
            match=match,
            valid_mask=valid_mask,
            cur_pos_aligned=cur_pos_aligned,
            detector_name="DINO-Upsampled",
            tar_img_changed=tar_changed
        )
