from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

class SurfaceVisual:
    def __init__(self, surface):
        self.surface = surface
        self.fig = None
        self.ax = None
    def show_vis_3d(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, auto_add_to_figure=False)
        self.fig.add_axes(self.ax)
        verts = []
        for triangle in self.surface.triangles:
            [x1,y1,z1] = triangle.vertices[0].c
            [x2,y2,z2] = triangle.vertices[1].c
            [x3, y3, z3] = triangle.vertices[2].c


            # [x1,y1,z1] = (action_matrix @ np.array([[x1], [y1], [z1]])).reshape(3,)
            # [x2,y2,z2] = (action_matrix @ np.array([[x2], [y2], [z2]])).reshape(3,)
            # [x3,y3,z3] = (action_matrix @ np.array([[x3], [y3], [z3]])).reshape(3,)
            


            [x1, y1, z1] = np.array([x1, y1, z1])/np.sum(np.array([x1, y1, z1]))
            [x2, y2, z2] = np.array([x2, y2, z2])/np.sum(np.array([x2, y2, z2]))
            [x3, y3, z3] = np.array([x3, y3, z3])/np.sum(np.array([x3, y3, z3]))

            x = [x1, x2, x3]
            y = [y1, y2, y3]
            z = [z1, z2, z3]
            verts.append(list(zip(x,y,z)))
        unique_indices = []
        for triangle in self.surface.triangles:
            unique_indices.append(triangle.index)
        unique_indices = list(set(unique_indices))
        choices = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:olive", "tab:brown", "tab:pink"]
        if len(unique_indices) > len(choices):
           choices = choices * int(np.ceil(len(self.surface.triangles) / len(choices)) * len(choices))
        colors = [choices[self.surface.triangles[index].index] for index in range(len(self.surface.triangles))]  # MWE colors
        patchcollection = Poly3DCollection(verts, linewidth=1, edgecolor="k", facecolor=colors, rasterized=True)
        # self.ax.add_collection3d(patchcollection)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

######################################
        
        action_matrix = np.array([[4.0, 4.0, 1.0],
                                [2.0, 3.0, 1.0],
                                [1.0, 2.0, 1.0]])
    

        eigenvectors = np.linalg.eig(action_matrix).eigenvectors
        fixed_point1 = eigenvectors[0]/np.sum(eigenvectors[0])
        fixed_point2 = eigenvectors[1]/np.sum(eigenvectors[1])
        fixed_point3 = eigenvectors[2]/np.sum(eigenvectors[2])

        x = [fixed_point1[0], fixed_point2[0], fixed_point3[0]]
        y = [fixed_point1[1], fixed_point2[1], fixed_point3[1]]
        z = [fixed_point1[2], fixed_point2[2], fixed_point3[2]]

        # self.ax.add_collection3d(Poly3DCollection([list(zip(x,y,z))], linewidth=1, edgecolor="k", facecolor=["tab:red"], rasterized=True))
        self.ax.scatter(fixed_point1[0], fixed_point1[1], fixed_point1[2], c='red', s=50, zorder=2)
        self.ax.scatter(fixed_point2[0], fixed_point2[1], fixed_point2[2], c='red', s=50, zorder=2)
        self.ax.scatter(fixed_point3[0], fixed_point3[1], fixed_point3[2], c='red', s=50, zorder=2)

######################################


        action_matrix2 = np.array([[0.0, 0.0, 1.0],
            [0.0, -1.0, -3.0],
            [1.0, 6.0, 9.0]])
        
        eigenvectors = np.linalg.eig(action_matrix2).eigenvectors
        fixed_point1 = eigenvectors[0]/np.sum(eigenvectors[0])
        fixed_point2 = eigenvectors[1]/np.sum(eigenvectors[1])
        fixed_point3 = eigenvectors[2]/np.sum(eigenvectors[2])

        x = [fixed_point1[0], fixed_point2[0], fixed_point3[0]]
        y = [fixed_point1[1], fixed_point2[1], fixed_point3[1]]
        z = [fixed_point1[2], fixed_point2[2], fixed_point3[2]]

        # self.ax.add_collection3d(Poly3DCollection([list(zip(x,y,z))], linewidth=1, edgecolor="k", facecolor=["tab:orange"], rasterized=True))
        self.ax.scatter(fixed_point1[0], fixed_point1[1], fixed_point1[2], c='orange', s=50, zorder=2)
        self.ax.scatter(fixed_point2[0], fixed_point2[1], fixed_point2[2], c='orange', s=50, zorder=2)
        self.ax.scatter(fixed_point3[0], fixed_point3[1], fixed_point3[2], c='orange', s=50, zorder=2)

            
        def alpha_inverse(x):
            [x,y] = x
            x_output = (x-1)/(2*(np.cos(2*np.pi/3)-1)) + y/(2*np.sin(2*np.pi/3))
            y_output = (x-1)/(2*(np.cos(2*np.pi/3)-1)) - y/(2*np.sin(2*np.pi/3))
            z_output = (np.cos(2*np.pi/3) - x)/(np.cos(2*np.pi/3) - 1)
            return [x_output, y_output, z_output]
        
        inverted_circle_pts = []
        for theta in np.linspace(0, 2*np.pi, 100):
            inverted_circle_pts.append(alpha_inverse([np.cos(theta), np.sin(theta)]))

        inverted_circle_pts = np.array(inverted_circle_pts)
        x = inverted_circle_pts[:,0]
        y = inverted_circle_pts[:, 1]
        z = inverted_circle_pts[:, 2]

        self.ax.plot(x,y,z, c='red')

        self.ax.set_box_aspect((1, 1, 1), zoom=0.5)

        plt.show()
    def show_vis_projected_3d(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, auto_add_to_figure=False)
        self.fig.add_axes(self.ax)
        verts = []



        all_vertices = []
        for triangle in self.surface.triangles:
            for vertex in triangle.vertices:
                if vertex not in all_vertices:
                    all_vertices.append(vertex)
        old_coords = []
        for vertex in all_vertices:
            old_coords.append(vertex.c)
            vertex.c = [vertex.c[0], vertex.c[1], vertex.c[2], 1]
            vertex.c = vertex.c / np.linalg.norm(vertex.c)
            vertex.c = np.array(vertex.c[1:]) / (1 + vertex.c[0])
        for triangle in self.surface.triangles:
            [x1, y1, z1] = triangle.vertices[0].c
            [x2, y2, z2] = triangle.vertices[1].c
            [x3, y3, z3] = triangle.vertices[2].c

            x = [x1, x2, x3]
            y = [y1, y2, y3]
            z = [z1, z2, z3]
            verts.append(list(zip(x, y, z)))
        vertex_index = 0
        for vertex in all_vertices:
            vertex.c = old_coords[vertex_index]
            vertex_index+=1

        unique_indices = []
        for triangle in self.surface.triangles:
            unique_indices.append(triangle.index)
        unique_indices = list(set(unique_indices))
        choices = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:olive", "tab:brown", "tab:pink"]
        if len(unique_indices) > len(choices):
            choices = choices * int(np.ceil(len(self.surface.triangles) / len(choices)) * len(choices))
        colors = [choices[self.surface.triangles[index].index] for index in
                  range(len(self.surface.triangles))]  # MWE colors
        patchcollection = Poly3DCollection(verts, linewidth=1, edgecolor="k", facecolor=colors, rasterized=True)
        self.ax.add_collection3d(patchcollection)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        plt.show()