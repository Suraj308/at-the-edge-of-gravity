import numpy as np
import matplotlib.pyplot as plt

data=np.load("data/spacetime_data.npy")

mass=data[:,0]
distance=data[:,1]

distortion=data[:,2]

#Habitability threshold CONCEPTUAL
SAFE_LIMIT = 0.02
RISK_LIMIT =0.08

habitability = np.zeros_like(distortion)

habitability[distortion<SAFE_LIMIT]=0
habitability[(distortion>=SAFE_LIMIT) & (distortion<RISK_LIMIT)]=1
habitability[distortion>=RISK_LIMIT]=2

#PLOT  IT
plt.figure(figsize=(8,15))
plt.scatter(distance[habitability==0],distortion[habitability==0],s=10,label="Habitable",alpha=0.6)
plt.scatter(distance[habitability==1],distortion[habitability==1],s=10,label="Marginal",alpha=0.6)
plt.scatter(distance[habitability==2],distortion[habitability==2],s=10,label="Unhabitable",alpha=0.6)

plt.axhline(SAFE_LIMIT,linestyle="--")
plt.axhline(RISK_LIMIT,linestyle="--")

plt.xlabel("Distance")
plt.ylabel("Spacetime Distortion")
plt.title("Habitable Zones in Curved Spacetime")
plt.legend()
plt.show()