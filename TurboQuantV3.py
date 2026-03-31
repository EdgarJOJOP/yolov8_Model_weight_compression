import torch
import torch.nn as nn
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# [保持 generate_rotation_matrix 和 LloydMaxCodebook 不变...]
def generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return Q * diag_sign.unsqueeze(0)

class YOLOv8TurboManager:
    def __init__(self, model, device="cuda"):
        self.device = torch.device(device)
        # --- 修复 1: 必须先融合 BN，否则量化后的权重在推理时是错的 ---
        print("Fusing model before compression...")
        self.model = model.fuse().to(self.device) 
        self.compressed_layers = {}

    def _get_protected_names(self):
        protected = []
        for name, m in self.model.named_modules():
            if isinstance(m, Detect):
                protected.append(name)
        return protected

    @torch.no_grad()
    def run_compression(self, base_bits=4):
        protected_names = self._get_protected_names()
        
        for idx, (name, m) in enumerate(self.model.named_modules()):
            if not isinstance(m, nn.Conv2d):
                continue
            
            # 保护检测头
            if any(p in name for p in protected_names):
                print(f"Skipping Quantization for Head: {name}")
                continue # 直接跳过，不存入 compressed_layers，推理时保持原样
            bits = 8 if name == "model.0" else base_bits # 仅保护第一层
            
            w = m.weight.data
            OC, IC, K1, K2 = w.shape
            head_dim = IC * K1 * K2
            flat_w = w.view(OC, head_dim).float()
            
            # 1. 归一化
            norms = torch.norm(flat_w, dim=-1, keepdim=True) + 1e-8
            flat_w_normed = flat_w / norms
            
            # 2. 旋转
            seed = idx + 1000
            Pi = generate_rotation_matrix(head_dim, seed, device=self.device)
            y = flat_w_normed.to(self.device) @ Pi.T.to(self.device)  # 旋转后的数据
            
            # --- 核心改进：计算动态 Scale ---
            # 找出旋转后数据的实际分布范围，这能防止数值爆炸
            scale = y.abs().max().item() 
            
            # 3. 生成质心（在 -scale 到 scale 之间分布）
            num_levels = 2 ** bits
            # 这里的 linspace 变成归一化的，推理时乘以 scale
            centroids = torch.linspace(-1, 1, num_levels).to(self.device)
            
            # 4. 寻找索引 (量化 y / scale)
            y_normed = y / (scale + 1e-8)
            diffs = (y_normed.unsqueeze(-1) - centroids).abs()
            indices = diffs.argmin(dim=-1).to(torch.uint8)
            
            # 5. 打包并存储
            packed_indices = self._pack_indices(indices, bits)
            self.compressed_layers[name] = {
                "packed_indices": packed_indices.cpu(),
                "norms": norms.half().cpu(),
                "scale": scale,           # 修复：必须保存 scale
                "bits": bits,
                "seed": seed,
                "head_dim": head_dim,
                "orig_shape": list(w.shape)
            }

    def _pack_indices(self, indices, bits):
        if bits == 8: return indices
        N, D = indices.shape
        # 确保 D 是偶数以便 4-bit 打包
        if D % 2 != 0:
            indices = torch.cat([indices, torch.zeros(N, 1, device=indices.device, dtype=torch.uint8)], dim=1)
        
        indices = indices.view(N, -1, 2).long()
        packed = (indices[:, :, 0] << 4) | (indices[:, :, 1])
        return packed.to(torch.uint8)

    def save(self, path: str):
        torch.save(self.compressed_layers, path)
        print(f"Saved to {path}")

if __name__ == "__main__":
    yolo = YOLO("yolov8n.pt")
    manager = YOLOv8TurboManager(yolo.model)
    manager.run_compression(base_bits=4)
    manager.save("yolov8n_turbo_packed.pth")
