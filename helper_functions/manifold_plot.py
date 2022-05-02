from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from add_new_triangle_functions import compute_translation_matrix_torus
import mpmath as mp
    
x = np.array([1,1,1,71.71067812,71.71067812,1,1,1])

alpha1,alpha2=compute_translation_matrix_torus(x)

#print(alpha1*mp.inverse(alpha1))


#print(mp.eig(alpha1)[0])
#print(mp.eig(alpha2)[0])








exit()
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


def theta1(V):
    V = np.array(V)
    result = np.array(V.copy())
    result[result > np.pi/2] = 0
    result[result < -np.pi/2] = 0
    result[np.logical_and(V <= np.pi/2, V >= -np.pi/2)] = np.arccos(4/(2*(2+np.cos(result[np.logical_and( V<= np.pi/2, V >= -np.pi/2)]))))
    return result

def u(X,V):
    result = np.array(X.copy())
    initial_theta = theta1(V)
    result[X<=1/2] = 4*(np.pi-initial_theta[X<=1/2])*result[X<=1/2] + initial_theta[X<=1/2]
    result[X> 1/2] = 4*(np.pi-initial_theta[X  > 1/2])*(result[X>1/2]-1/2)+np.pi-initial_theta[X  > 1/2]
    return result




def r(PHI,THETA):
    return np.array([(2+np.cos(PHI))*np.cos(THETA),(2+np.cos(PHI))*np.sin(THETA),np.sin(PHI)])

def r2(X,V):

    results = np.zeros(shape=(3,X.shape[0],V.shape[1]))

    results[:,X<=1/2] = r(u(X[X<=1/2],V[X<=1/2]),V[X<=1/2])
    X_pos, Y_pos, Z_pos = r(u(X[X>1/2],V[X>1/2]),V[X>1/2])
    results[:,X>1/2] = np.array([X_pos+4,Y_pos,Z_pos])

    # results[X<=1/2] = r(u(X[X<=1/2],V[X<=1/2]),V[X<=1/2])
    # 
    # results[X>1/2] = np.array([X_pos+3,Y_pos,Z_pos])
    return results


def f(X,Y):
    return r(2*np.pi*X,2*np.pi*Y)



# x,v= np.meshgrid(np.linspace(0,1,100),np.linspace(-np.pi,np.pi, 100))

# X,Y,Z = r2(x,v)

# plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot_surface(X,Y,Z)
# plt.show()




X, Y, Z = r(PHI,THETA)


fig = plt.figure()



ax = plt.axes(projection="3d")

for ii in np.linspace(0,360,360):
    ax.plot_surface(X,Y,Z,color='white',alpha=0.5)

    # X = np.linspace(0,1,100)
    # X, Y, Z = f(X,X)
    # ax.plot3D(X,Y,Z,color='green')

    X = np.linspace(0,1,200)
    X, Y, Z = f(X,0*X)
    ax.plot3D(X,Y,Z,color='red')

    X = np.linspace(0,1,200)
    X, Y, Z = f(0*X,X)
    ax.plot3D(X,Y,Z,color='blue')


    ax.set_zlim(-4,4)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_axis_off()

    ax.view_init(elev=10., azim=ii)
    plt.savefig("~/Documents/University/Dalyell Project/Presentation/torus_rotate/movie%d.png" % ii)


plt.show()



# X = np.linspace(0,10,50)
# Y = np.sin(X)
# Z = X**2 + Y**2

# fig = plt.figure()
# norm = mpl.colors.Normalize(vmin=0, vmax=max(Z))
# cmap = cm.jet
# m = cm.ScalarMappable(norm=norm, cmap=cmap)
# ax = fig.add_subplot(1,1,1,projection='3d')
# N = len(X)
# for i in range(N-1):
#     ax.plot(X[i:i+2],Y[i:i+2],Z[i:i+2],color=plt.cm.jet(1/2*(Z[i]+Z[i+1])/max(Z)))

# # mpl.colorbar.ColorbarBase(ax, cmap=cmap,
# #                                 norm=norm,
# #                                 orientation='horizontal')
# m._A = []
# fig.colorbar(m)
# plt.show()



    