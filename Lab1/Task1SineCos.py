import torch
import numpy as np

print("Pytorch Version: ", torch.__version__)

#device configuration
device = torch.device("mps")

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.Tensor(Y)

x = x.to(device)
y = y.to(device)


#compute Gaussian
z = torch.sin(y+x) 

#plot
import matplotlib.pyplot as plt
plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()


