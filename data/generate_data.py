import numpy as np
import matplotlib.pyplot as plt
#no. of points
N=6000

#Values for mass Random
mass=np.random.uniform(1.0,10.0,N)

#Random values for ditance with wider range as distance affect gravity more than mass
distance=np.random.uniform(1.0,60.0,N)


#Simple graity like distortion Formula
#This is a simplified, Physics -inspired proxy
distortion = mass/(distance**2)

#All values in Column
data=np.column_stack((mass,distance,distortion))

#Save File
np.save("data/spacetime_data.npy",data)

#Plot to check
plt.scatter(distance[:500],distortion[:500],s=5)
plt.xlabel("Distance")
plt.ylabel("Space Distortion")
plt.title("Distortion with respect to Distance")
plt.show()
print(distortion.min())