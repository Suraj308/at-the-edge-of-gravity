import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data=np.load("data/spacetime_data.npy")

mass=data[:,0]
distance=data[:,1]

true_distortion=data[:,2]

class spacetimenet(nn.Module):
    def __init__(self):
        super() .__init__()
        self.model =nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.model(x)
    
model=spacetimenet()
model.load_state_dict(torch.load("model/spacetime_model.pt"))
model.eval()

X=np.column_stack((mass,distance))
X_mean=X.mean(axis=0)
X_std = X.std(axis=0)

X_norm = (X-X_mean)/X_std

X_tensor = torch.tensor(X_norm , dtype=torch.float32)

with torch.no_grad():
    predicted_distortion=model(X_tensor).numpy().flatten()

plt.figure(figsize=(8,5))
plt.scatter(distance[:500],true_distortion[:500],s=10, label="Physics",alpha=0.6 )
plt.scatter(distance[:500],predicted_distortion[:500],s=10,label="ML Prediction",alpha=0.6)

plt.xlabel("Distance")
plt.ylabel("Spacetime Distortion")
plt.title("Physics vs Neural Network Prediction")
plt.legend()
plt.show()