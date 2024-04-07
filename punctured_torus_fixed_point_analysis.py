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



r3, c3 = compute_all_until_r3c3(r0, r2, c0, c2, e03, e23, e30, e32, t_prime)

r4, c4 = compute_all_until_r3c3(r3, r2, c3, c2, e34, e24, e43, e42, t)

def alpha_inverse(x):
        [x,y] = x
        x_output = (x-1)/(2*(np.cos(2*np.pi/3)-1)) + y/(2*np.sin(2*np.pi/3))
        y_output = (x-1)/(2*(np.cos(2*np.pi/3)-1)) - y/(2*np.sin(2*np.pi/3))
        z_output = (np.cos(2*np.pi/3) - x)/(np.cos(2*np.pi/3) - 1)
        return [x_output, y_output, z_output]

def get_matrix_representation(A,B,C,A_prime,B_prime,C_prime):
    """Finds the matrix representation f : R^3 -> R^3 with contraints
        f(A) = A'
        f(B) = B'
        f(C) = C'
        where (A,B,C) and (A',B',C') are bases of R^3, represented with respect to the standard basis"""
    
    e1 = [1,0,0]
    e2 = [0,1,0]
    e3 = [0,0,1]

    left_matrix_first_row = np.concatenate([A[0]*np.identity(3), A[1]*np.identity(3), A[2]*np.identity(3)], axis=1)
    left_matrix_second_row = np.concatenate([B[0]*np.identity(3), B[1]*np.identity(3), B[2]*np.identity(3)], axis=1)
    left_matrix_third_row = np.concatenate([C[0]*np.identity(3), C[1]*np.identity(3), C[2]*np.identity(3)], axis=1)
    left_matrix = np.concatenate([left_matrix_first_row, left_matrix_second_row, left_matrix_third_row], axis=0)

    right_matrix_first_row = np.concatenate([A_prime[0]*np.identity(3), A_prime[1]*np.identity(3), A_prime[2]*np.identity(3)], axis=1)
    right_matrix_second_row = np.concatenate([B_prime[0]*np.identity(3), B_prime[1]*np.identity(3), B_prime[2]*np.identity(3)], axis=1)
    right_matrix_third_row = np.concatenate([C_prime[0]*np.identity(3), C_prime[1]*np.identity(3), C_prime[2]*np.identity(3)], axis=1)
    right_matrix = np.concatenate([right_matrix_first_row, right_matrix_second_row, right_matrix_third_row], axis=0)

    components_matrix = np.linalg.inv(left_matrix) @ right_matrix @ np.array([e1,e2,e3]).reshape(9,1)
    return components_matrix.reshape(3,3).T

def get_fixed_points_on_affine_chart(matrix):
    v1, v2, v3 = np.linalg.eig(matrix).eigenvectors
    v1, v2, v3 = v1/np.sum(v1), v2/np.sum(v2), v3/np.sum(v3)
    return v1, v2, v3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

triangle1 = np.array([c0/np.sum(c0), c1/np.sum(c1), c2/np.sum(c2), c0/np.sum(c0)])
triangle2 = np.array([c0/np.sum(c0), c2/np.sum(c2), c3/np.sum(c3), c0/np.sum(c0)])
triangle3 = np.array([c3/np.sum(c3), c2/np.sum(c2), c4/np.sum(c4), c3/np.sum(c3)])
points = np.array([c0/np.sum(c0), c1/np.sum(c1), c2/np.sum(c2), c3/np.sum(c3), c4/np.sum(c4)])

ax.scatter(points[:,0], points[:,1], points[:,2])
ax.plot(triangle1[:,0], triangle1[:,1], triangle1[:,2], c='red')
ax.plot(triangle2[:,0], triangle2[:,1], triangle2[:,2], c='blue')
ax.plot(triangle3[:,0], triangle3[:,1], triangle3[:,2], c='green')

alpha = get_matrix_representation(c0, c1, c2, c3, c2, c4)
v1,v2,v3 = get_fixed_points_on_affine_chart(alpha)

# print(alpha @ np.array(c2).reshape(3,1) - np.array(c4).reshape(3,1))


fixed_points = np.array([v1,v2,v3])

inverted_circle_pts = []
for theta in np.linspace(0, 2*np.pi, 100):
    inverted_circle_pts.append(alpha_inverse([np.cos(theta), np.sin(theta)]))

inverted_circle_pts = np.array(inverted_circle_pts)

ax.scatter(fixed_points[:,0],fixed_points[:,1],fixed_points[:,2], c='green')

ax.plot(inverted_circle_pts[:,0],inverted_circle_pts[:,1],inverted_circle_pts[:,2], c='red')


plt.show()

