

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from ultralytics import YOLO

# [CUDA 内核部分保持不变...]
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void unpack_4bit_to_float_kernel(
    const uint8_t* __restrict__ packed_indices,
    const float* __restrict__ centroids,
    float* __restrict__ output,
    int N, int D, int n_groups) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D;
    if (idx >= total_elements) return;

    int row = idx / D;
    int col = idx % D;
    int byte_idx = col / 2;
    int bit_pos = (col % 2 == 0) ? 4 : 0; 
  
    uint8_t packed_val = packed_indices[row * n_groups + byte_idx];
    uint8_t centroid_idx = (packed_val >> bit_pos) & 0xF;
    output[idx] = centroids[centroid_idx];
}

torch::Tensor unpack_4bit(torch::Tensor packed_indices, torch::Tensor centroids, int D) {
    int N = packed_indices.size(0);
    int n_groups = packed_indices.size(1);
    auto output = torch::empty({N, D}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const int threads = 256;
    const int blocks = (N * D + threads - 1) / threads;
    unpack_4bit_to_float_kernel<<<blocks, threads>>>(
        (const uint8_t*)packed_indices.data_ptr(), centroids.data_ptr<float>(), output.data_ptr<float>(), N, D, n_groups
    );
    return output;
}
"""
cpp_source = "torch::Tensor unpack_4bit(torch::Tensor packed_indices, torch::Tensor centroids, int D);"
tq_cuda = load_inline(name='tq_v4', cpp_sources=cpp_source, cuda_sources=cuda_source, functions=['unpack_4bit'], with_cuda=True,extra_cuda_cflags=['-allow-unsupported-compiler', '--use_fast_math'])

class TurboQuantFusedConv(nn.Module):
    def __init__(self, original_conv, bundle):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = original_conv.stride, original_conv.padding, original_conv.dilation, original_conv.groups
        self.bias = original_conv.bias 
        self.bits = bundle["bits"]
        self.D = bundle["head_dim"]
        self.original_shape = bundle["orig_shape"]

        self.register_buffer("packed_indices", bundle["packed_indices"].cuda())
        self.register_buffer("norms", bundle["norms"].cuda().float().view(-1, 1))
        self.scale = bundle["scale"] # 读取缩放因子
        self.bits = bundle["bits"]
        
        # 重新生成旋转矩阵 Pi 和质心
        num_levels = 2 ** self.bits
        self.register_buffer("centroids", torch.linspace(-1, 1, num_levels).cuda().float())
        # 重新生成旋转矩阵 Pi
        pi = self._get_rotation(self.D, bundle["seed"])
        self.register_buffer("pi_matrix", pi.cuda().float())
    def _get_rotation(self, d, seed):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(d, d, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        return Q * diag_sign.unsqueeze(0)

    @torch.no_grad()
    def forward(self, x):
        if self.bits == 4:
            actual_D = self.packed_indices.shape[1] * 2
            u = tq_cuda.unpack_4bit(self.packed_indices, self.centroids, actual_D)[:, :self.D]
        else:
            u = self.centroids[self.packed_indices.long()]
      
        # --- 核心数学修正 ---
        # W = (U * Scale @ Pi) * Norm
        # 先应用 scale，再乘以旋转矩阵
        w_flat = torch.mm(u * self.scale, self.pi_matrix) * self.norms
        
        w = w_flat.view(self.original_shape).to(x.dtype)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def convert_to_turbo(model, compressed_path):
    # 预测前也必须 fuse，确保结构一致
    model = model.fuse() 
    compressed_dict = torch.load(compressed_path)
    count = 0
    for name, m in model.named_modules():
        if name in compressed_dict:
            parts = name.split('.')
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            setattr(parent, parts[-1], TurboQuantFusedConv(m, compressed_dict[name]))
            count += 1
    print(f"Successfully injected {count} TurboQuant layers.")
    return model

if __name__ == "__main__":
    yolo = YOLO("yolov8n.pt")
    yolo.model = convert_to_turbo(yolo.model, "yolov8n_turbo_packed.pth")
    yolo.model.cuda().eval()
    import time
    # 运行预测
    for _ in range(20):
        results = yolo.predict("https://ultralytics.com/images/bus.jpg", conf=0.15, device="cuda", save=True)
    for result in results:
        print(f"Detected {len(result.boxes)} objects.")
    # 6. 正式推理性能测试
    print("开始推理测试...")
    iterations = 100
    start_time = time.time()

    for i in range(iterations):
        # verbose=False 禁用控制台打印以获取纯粹的推理时间
        results = yolo.predict("bus.jpg", device=0, verbose=False)

    end_time = time.time()
    
    # 7. 计算结果
    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # 毫秒
    fps = 1.0 / (avg_time / 1000)

    print("-" * 30)
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"推理帧率 (FPS): {fps:.2f}")
    print("-" * 30)
