from decorated_triangles.triangle import *
from visualise.surface_vis import SurfaceVisual
import time

my_surface = Surface(2,1,Triangle(Decoration([1,2,1],[1,2,3],[1,4,1])))
my_surface.add_vertex(my_surface.triangles[-1],[2,-1,3])
my_surface.add_vertex(my_surface.triangles[-1],[2,0,2])
surface_vis = SurfaceVisual(my_surface)
surface_vis.show_vis()



# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
#     #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
