"""Compare CPU and original-device float64 Sinkhorn pair sampling."""
import torch
from flow import ot_pairs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for CPU/GPU solver comparison")
    torch.set_num_threads(2)
    for seed in range(100):
        torch.manual_seed(seed)
        source = torch.randn(32, 288, device="cuda")
        target = torch.randn(32, 288, device="cuda")
        labels = torch.arange(32, device="cuda") % 7
        q = torch.softmax(torch.randn(32, 7, device="cuda"), dim=1)
        cpu_rng = torch.Generator().manual_seed(seed + 10000)
        gpu_rng = torch.Generator().manual_seed(seed + 10000)
        a = ot_pairs(source, target, labels, q, cpu_rng, solver_device="cpu")
        b = ot_pairs(source, target, labels, q, gpu_rng, solver_device="same")
        assert all(torch.equal(x, y) for x, y in zip(a, b)), seed
        assert torch.equal(cpu_rng.get_state(), gpu_rng.get_state()), seed
    print("100/100 identical OT pair indices and sampler RNG states")


if __name__ == "__main__":
    main()
