from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
x1 = [1,0,1]
y1 = [0,1,1]
z1 = [0,0,1]
x2 = [1,1,4]
y2 = [0,1,1]
z2 = [0,1,2]
verts = [list(zip(x1,y1,z1)),list(zip(x2,y2,z2))]
colors = ["tab:blue" if np.random.rand()<0.1 else "tab:orange" for vert in verts]  # MWE colors
print(colors)
patchcollection = Poly3DCollection(verts,linewidth=1,edgecolor="k",facecolor = colors,rasterized=True)
ax.add_collection3d(patchcollection)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.add_collection3d(Poly3DCollection(verts))
plt.show()