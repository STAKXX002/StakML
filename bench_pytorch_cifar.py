"""
Benchmarks PyTorch CPU eager-mode forward+backward for the exact same
layer shapes StakML's cifar_cnn.cpp uses, at the same batch size.

Compare the printed per-step numbers directly against StakML's existing
[timing] instrumentation output (data/fwd/bwd/opt.step per epoch, divide
by 781 steps to get per-step numbers).

Run: python3 bench_pytorch_cifar.py
"""
import time
import torch
import torch.nn as nn

torch.set_num_threads(16)  # match StakML's default OpenMP thread count first;
                            # re-run with torch.set_num_threads(8) afterward
                            # to test whether SMT siblings help or hurt here

BATCH = 64
STEPS = 100  # enough for a stable average; StakML does 781/epoch

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(BATCH, 3, 32, 32)
y = torch.randint(0, 10, (BATCH,))

# warmup — first few iterations pay one-time allocator/cache costs
for _ in range(10):
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()

t_fwd = t_bwd = t_opt = 0.0
for _ in range(STEPS):
    opt.zero_grad()

    ta = time.perf_counter()
    out = model(x)
    tb = time.perf_counter()

    loss = loss_fn(out, y)
    loss.backward()
    tc = time.perf_counter()

    opt.step()
    td = time.perf_counter()

    t_fwd += tb - ta
    t_bwd += tc - tb
    t_opt += td - tc

print(f"threads={torch.get_num_threads()}  steps={STEPS}")
print(f"avg fwd/step:      {1000*t_fwd/STEPS:.2f} ms")
print(f"avg bwd/step:      {1000*t_bwd/STEPS:.2f} ms")
print(f"avg opt.step/step: {1000*t_opt/STEPS:.2f} ms")
print(f"avg total/step:    {1000*(t_fwd+t_bwd+t_opt)/STEPS:.2f} ms")
print(f"projected epoch (781 steps): {(t_fwd+t_bwd+t_opt)/STEPS*781:.1f}s (compute only, no data loading/eval)")
