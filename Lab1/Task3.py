import torch
import matplotlib.pyplot as plt


def koch_snowflake(iterations, length=1.0):
    # Initial triangle vertices
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([length, 0.0])
    p2 = torch.tensor([length / 2.0, length * torch.sqrt(torch.tensor(3.0)) / 2.0])

    points = [p0, p1, p2, p0]  # Closing the loop by adding p0 again at the end

    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p_start = points[i]
            p_end = points[i + 1]
            # The vector between two consecutive points
            vector = p_end - p_start
            # Divide the vector into 3 parts
            p1 = p_start + vector / 3
            p2 = p_start + 2 * vector / 3
            # Create the peak of the bump
            angle = torch.atan2(vector[1], vector[0])
            length = torch.norm(vector) / 3
            peak = p1 + torch.tensor([torch.cos(angle - torch.pi / 3) * length,
                                      torch.sin(angle - torch.pi / 3) * length])
            # Add the new segments
            new_points.extend([p_start, p1, peak, p2])
        new_points.append(points[-1])  # Append last point
        points = new_points

    return torch.stack(points)



# Set up the iterations and draw the fractal
iterations = 5
points = koch_snowflake(iterations)

# Plot the Koch snowflake
plt.figure(figsize=(8, 8))
plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), color='blue')
plt.fill(points[:, 0].numpy(), points[:, 1].numpy(), color='cyan', alpha=0.5)
plt.axis('equal')
plt.axis('off')
plt.title(f"Koch Snowflake - Iterations: {iterations}")
plt.show()