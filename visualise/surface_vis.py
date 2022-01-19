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
            [x1,y1,z1] = triangle.edges[0].v0.c
            [x2,y2,z2] = triangle.edges[0].v1.c
            [x3, y3, z3] = triangle.edges[1].v1.c
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