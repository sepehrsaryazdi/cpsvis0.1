from helper_functions.add_new_triangle_functions import compute_all_until_r3c3
import matplotlib.pyplot as plt
import numpy as np

e01 = 1
e02 = 1
e10 = 1
e12 = 1
e20 = 1
e21 = 1
t = 1

e03 = e12
e30 = e21
e32 = e01
e23 = e10
e34 = e02
e43 = e20
e24 = e12
e42 = e21


t_prime = 1


c0 = [1,0,0]
c1 = [0,t,0]
c2 = [0,0,1]
r0 = [0, e01 / t, e02]
r1= [e10, 0, e12]
r2= [e20, e21 / t, 0]
c0_clover = [1, 0, 0]
c1_clover = [0, 1, 0]
c2_clover = [0, 0, 1]


r3, c3 = compute_all_until_r3c3(r0, r2, c0, c2, e03, e23, e30, e32, t_prime)

r4, c4 = compute_all_until_r3c3(r3, r2, c3, c2, e34, e24, e43, e42, t)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

triangle1 = np.array([c0, c1, c2, c0])
triangle2 = np.array([c0, c2, c3, c0])
triangle3 = np.array([c3, c2, c4, c3])
points = np.array([c0, c1, c2, c3, c4])

ax.scatter(points[:,0], points[:,1], points[:,2])
ax.plot(triangle1[:,0], triangle1[:,1], triangle1[:,2], c='red')
ax.plot(triangle2[:,0], triangle2[:,1], triangle2[:,2], c='blue')
ax.plot(triangle3[:,0], triangle3[:,1], triangle3[:,2], c='green')

plt.show()

