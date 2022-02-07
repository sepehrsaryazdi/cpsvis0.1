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
            x = [x1, x2, x3]
            y = [y1, y2, y3]
            z = [z1, z2, z3]
            verts.append(list(zip(x,y,z)))
        choices = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        colors = [choices[np.random.randint(0,5)] for vert in verts]  # MWE colors
        patchcollection = Poly3DCollection(verts, linewidth=1, edgecolor="k", facecolor=colors, rasterized=True)
        self.ax.add_collection3d(patchcollection)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
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
        choices = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        colors = [choices[np.random.randint(0, 5)] for vert in verts]  # MWE colors
        patchcollection = Poly3DCollection(verts, linewidth=1, edgecolor="k", facecolor=colors, rasterized=True)
        self.ax.add_collection3d(patchcollection)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        plt.show()