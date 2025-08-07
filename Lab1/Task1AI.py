import torch
import matplotlib.pyplot as plt

# --- Gaussian parameters ----------------------------------------------------
mu_x, mu_y   = 0.0, 0.0     # peak centre
sigma_x      = 1.0          # spread in x
sigma_y      = 1.0          # spread in y
# ----------------------------------------------------------------------------

def gaussian_2d(x, y, mux, muy, sigx, sigy):
    """2-D Gaussian evaluated on PyTorch tensors."""
    return torch.exp(-(((x - mux) ** 2) / (2 * sigx ** 2) +
                       ((y - muy) ** 2) / (2 * sigy ** 2)))

# Build meshgrid with PyTorch (stays on default device ⇒ CPU or CUDA)
x = torch.linspace(-3, 3, 200)
y = torch.linspace(-3, 3, 200)
X, Y = torch.meshgrid(x, y, indexing='xy')     # Cartesian layout

Z = gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)

# Plot (convert to NumPy only for Matplotlib)
fig, ax = plt.subplots()
cf = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy())
plt.colorbar(cf, ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D Gaussian Function (PyTorch)")
plt.show()