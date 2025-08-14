import torch
import numpy as np
import matplotlib.pyplot as plt

def koch_antisnow(iterations, length = 1):
    x = torch.tensor([0.0,0.0])
    y = torch.tensor([length, 0.0])
    z = torch.tensor([length/2.0, length * torch.sqrt(torch.tensor(3.0)) / 2.0])
    
    points = [x,y,z,x]
    for _ in range(iterations):
        newpoints = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i+1]
            vector = end - start
            
            length = torch.norm(vector)/3
            angle = torch.atan2(vector[1], vector[0])
            
            p0 = start + vector/3.0
            p2 = end - vector/3.0
            p1 = p0 + torch.tensor([torch.cos(angle + torch.pi / 3) * length,
                                      torch.sin(angle + torch.pi / 3) * length])
            
            newpoints.extend([start, p0, p1, p2, end])
        newpoints.append(points[-1])
        points = newpoints
    return torch.stack(points)


# Set up the iterations and draw the fractal
iterations = 12
points = koch_antisnow(iterations)

# Plot the Koch snowflake
plt.figure(figsize=(8, 8))
plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), color='blue')
plt.fill(points[:, 0].numpy(), points[:, 1].numpy(), color='cyan', alpha=0.5)
plt.axis('equal')
plt.axis('off')
plt.title(f"Koch Snowflake - Iterations: {iterations}")
plt.show()


