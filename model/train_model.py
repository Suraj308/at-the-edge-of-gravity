import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

data= np.load("data/spacetime_data.npy")


#Input
mass = data[:,0]
distance=data[:,1]

#output
distortion = data[:,2]

X=np.column_stack((mass,distance))

y =distortion.reshape(-1,1)



X_mean = X.mean(axis=0)
X_std= X.std(axis=0)

X_norm = (X-X_mean)/X_std


#Convert numpy to tensor
X_tensors=torch.tensor(X_norm,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)

class spacetimenet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        
    def forward(self,x):
         return self.model(x)
        
model = spacetimenet()

criterion =nn.MSELoss()

optimiser =optim.Adam(model.parameters(),lr=0.01)

epochs=1000

for epoch in range(epochs):
    optimiser.zero_grad()

    prediction=model(X_tensors)

    loss = criterion(prediction,y_tensor)

    loss.backward()
    optimiser.step()

    if epoch%100==0:
        print(f"Epoch {epoch} , Loss: {loss.item():.6f}")

torch.save(model.state_dict(),"model/spacetime_model.pt")
