from triangle_class.decorated_triangle import *
from helper_functions.add_new_triangle_functions import *

t = 1
e01 = 2
e02 = 3
e10 = 4
e12 = 5
e20 = 6
e21 = 7

c0 = [1, 0, 0]
c1 = [0, t, 0]
c2 = [0, 0, 1]
r0 = [0, e01 / t, e02]
r1 = [e10, 0, e12]
r2 = [e20, e21 / t, 0]
c0_clover = [1, 0, 0]
c1_clover = [0, 1, 0]
c2_clover = [0, 0, 1]
x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
cube_root_x_coord_t = np.power(x_coord_t, 1 / 3)
r0_clover = [0, cube_root_x_coord_t, 1]
r1_clover = [1, 0, cube_root_x_coord_t]
r2_clover = [cube_root_x_coord_t, 1, 0]

e03 = 8
e30 = 9
e23 = 10
e32 = 11
A023 = 12

test_surface = Surface(c0, c1, c2, r0, r1, r2, c0_clover, c1_clover, c2_clover, r0_clover, r1_clover, r2_clover)


#
for triangle in test_surface.triangles:
    for edge in triangle.edges:
        print(edge.v0.c_clover)

print('e03: ', e03)
print('e30: ', e30)
print('e23: ', e23)
print('e32: ', e32)

r3,c3 = compute_all_until_r3c3(r0,r2,c0,c2,e03,e23,e30,e32,A023)
r3_clover,c3_clover = compute_all_until_r3c3(r0_clover,r2_clover,c0_clover,c2_clover,e03,e23,e30,e32,A023)


connecting_edge = test_surface.triangles[0].edges[0]

test_surface.add_triangle(connecting_edge, Vertex(c3,r3,c3_clover, r3_clover))

print(test_surface.triangles[1])