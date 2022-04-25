import numpy as np
import matplotlib.pyplot as plt


a = 4

# def theta(u):
    
#     theta1 = np.arccos((a-4)/2)
#     theta2 = 2*np.pi - theta1
#     if u > 1:
#         return theta2
#     elif u < 0:
#         return theta1
#     if u <= 1/2 and u >= 0:
#         return 2*(theta2-theta1)*u + theta1
#     if u > 1/2 and u <= 1:
#         return 2*(theta2-theta1)*u+np.pi - theta2

# def phi(v):
    
#     phi1 = np.arccos((a-4)/2)
#     phi2 = 2*np.pi - phi1
#     if v > 1:
#         return phi2
#     elif v < 0:
#         return phi1
#     if v <= 1/2 and v >= 0:
#         return 2*(phi2-phi1)*v + phi1
#     if v > 1/2 and v <= 1:
#         return 2*(phi2-phi1)*v+np.pi - phi2

def theta(u):
    return u*2*np.pi
def phi(v):
    return v*2*np.pi


U,V = np.meshgrid(np.linspace(0,1,50),np.linspace(0,1,50))

THETA = [[theta(U[i,j]) for j in range(len(U))] for i in range(len(U))]
PHI = [[phi(V[i,j]) for j in range(len(V))] for i in range(len(V))]

X = (2+np.cos(PHI))*np.cos(THETA)
Y = (2+np.cos(PHI))*np.sin(THETA)
Z = np.sin(PHI)


plt.figure()

ax = plt.axes(projection="3d")
ax.plot_surface(X-2,Y,Z)
ax.set_zlim(-2.5,2.5)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.plot_surface(X+2,Y,Z)
ax.set_axis_off()
plt.show()



    