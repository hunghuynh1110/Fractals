import torch
import matplotlib.pyplot as plt

device = torch.device("mps")
print("Running on:", device)


@torch.no_grad()
def koch_snowflake_vectorized(iterations: int, length: float = 1.0) -> torch.Tensor:
    """Fully vectorized Koch snowflake on device: all edges processed in parallel."""
    dtype = torch.float32

    # Build the initial equilateral triangle directly on device
    h = (torch.sqrt(torch.tensor(3.0, device=device, dtype=dtype)) / 2.0) * length
    p0 = torch.tensor([0.0,     0.0],     device=device, dtype=dtype)
    p1 = torch.tensor([length,  0.0],     device=device, dtype=dtype)
    p2 = torch.tensor([length/2.0, h],    device=device, dtype=dtype)

    # Use stack to combine existing tensors
    points = torch.stack([p0, p1, p2, p0], dim=0)

    for _ in range(iterations):
        # All edges in parallel
        p_start = points[:-1]                    # (N, 2)
        p_end   = points[ 1:]                    # (N, 2)
        v = p_end - p_start                      # (N, 2)

        p1 = p_start + v / 3.0                   # (N, 2)
        p2 = p_start + 2.0 * v / 3.0             # (N, 2)

        angles = torch.atan2(v[:, 1], v[:, 0])   # (N,)
        seg_len = torch.linalg.norm(v, dim=1) / 3.0  # (N,)

        dx = torch.cos(angles - torch.pi/3) * seg_len
        dy = torch.sin(angles - torch.pi/3) * seg_len
        peak = p1 + torch.stack([dx, dy], dim=1) # (N, 2)

        # Interleave [p_start, p1, peak, p2] for each edge in drawing order
        quad = torch.stack([p_start, p1, peak, p2], dim=1)  # (N, 4, 2)
        points = torch.cat([quad.reshape(-1, 2), points[-1:].clone()], dim=0)  # (4N+1, 2)

    return points


# Set up the iterations and draw the fractal
iterations = 6
points = koch_snowflake_vectorized(iterations)

#move points back to cpu before ploting them out
points = points.to(torch.device("cpu"))
# Plot the Koch snowflake
plt.figure(figsize=(8, 8))
plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), color='blue')
plt.fill(points[:, 0].numpy(), points[:, 1].numpy(), color='cyan', alpha=0.3)
plt.axis('equal')
plt.axis('off')
plt.title(f"Koch Snowflake - Iterations: {iterations}")
plt.show()