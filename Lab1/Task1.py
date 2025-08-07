import torch
import numpy as np

print("Pytorch Version: ", torch.__version__)

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.tensor(Y)

#transfer to the GPU device
x = x.to(device)
y = y.to(device)

#compute Gaussian
z = torch.sin(x) * torch.cos(y)

#plot
import matplotlib.pyplot as plt
plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()


