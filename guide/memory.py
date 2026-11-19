import torch
import torch.nn as nn

assert torch.cuda.is_available(), "需要 GPU"
device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()

# batch 要足够大，让激活值成为显存主项。混合精度不会缩小 FP32 权重和优化器状态，
# batch 太小时这部分固定开销会完全掩盖激活值的收益，甚至出现"负节省"
BATCH = 65536
data = torch.randn(BATCH, 1024, device=device)
labels = torch.randint(0, 10, (BATCH,), device=device)


def make_model():
    return nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device)


def step_peak(amp_dtype=None):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if amp_dtype is None:
        loss = criterion(model(data), labels)
    else:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = criterion(model(data), labels)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    # 关键：必须真正释放本轮的模型和优化器状态，否则会叠加到下一轮的峰值里
    del model, opt, loss
    torch.cuda.empty_cache()
    return peak


fp32_peak = step_peak(None)
bf16_peak = step_peak(torch.bfloat16)

print(
    f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | 节省: {(1-bf16_peak/fp32_peak)*100:.1f}%"
)
# 参考输出：FP32: 1881.6 MB | BF16: 1246.6 MB | 节省: 33.7%
