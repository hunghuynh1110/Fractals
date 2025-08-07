import torch
import matplotlib.pyplot as plt

# ---------------- parameters -----------------
img_width, img_height = 800, 800   # pixel resolution
max_iter = 100                     # iterations per point
re_min, re_max = -2.0, 1.0
im_min, im_max = -1.5, 1.5
# --------------------------------------------

# Build complex grid with PyTorch
re = torch.linspace(re_min, re_max, img_width)
im = torch.linspace(im_min, im_max, img_height)
X, Y = torch.meshgrid(re, im, indexing="xy")
C = torch.complex(X, Y)

# Initialise Z and bookkeeping tensors
Z = torch.zeros_like(C)
iter_counts = torch.zeros(C.shape, dtype=torch.int32)
active = torch.ones(C.shape, dtype=torch.bool)  # points still iterating

# Mandelbrot iteration loop
for i in range(max_iter):
    Z[active] = Z[active] * Z[active] + C[active]
    diverged = torch.abs(Z) > 2.0
    newly_diverged = diverged & active
    iter_counts[newly_diverged] = i
    active &= ~diverged
    if not active.any():
        break

# Colour in-set points with max_iter
iter_counts[active] = max_iter

# ---------- plot ----------
plt.figure(figsize=(6, 6))
plt.imshow(iter_counts.numpy().T,
           extent=[re_min, re_max, im_min, im_max],
           origin="lower")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Mandelbrot Set (PyTorch)")
plt.show()