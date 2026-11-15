"""Parameter counts, MACs/FLOPs and device latency."""
import time
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def compute_cost(model, image_size=64, device="cpu"):
    """Return THOP MACs and approximate FLOPs (2 * MACs), if installed."""
    try:
        from thop import profile
    except ImportError:
        return {"macs": None, "flops_approx": None}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(torch.randn(1, 3, image_size, image_size, device=device),), verbose=False)
        return {"macs": int(macs), "flops_approx": int(2 * macs)}
    finally:
        model.train(was_training)

def measure_latency(model, image_size=64, device="cpu", warmup=20, repeats=100):
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            sync()
            start = time.perf_counter()
            for _ in range(repeats):
                model(x)
            sync()
        return 1000 * (time.perf_counter() - start) / repeats
    finally:
        model.train(was_training)
