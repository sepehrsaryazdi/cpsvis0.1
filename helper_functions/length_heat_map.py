import matplotlib.pyplot as plt
from more_itertools import difference
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from helper_functions.add_new_triangle_functions import *
import mpmath as mp
mp.mp.dps = 300
mp.mp.pretty = False


class Node:
    def __init__(self,parent=None):
        self.parent = parent


class LengthHeatMapTree:
    def __init__(self,depth, ratio=1/2, alpha1= mp.matrix([[1,0,0],[0,1,0],[0,0,1]]), alpha2=mp.matrix([[1,0,0],[0,1,0],[0,0,1]]), difference_precision = 0.0001, k =2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.depth = depth
        self.ratio = ratio
        self.k = k
        initial_node = Node()
        initial_node.index = ''
        initial_node.coord = np.array([0,0])
        initial_node.matrix = mp.matrix([[1,0,0],[0,1,0],[0,0,1]])
        initial_node.length = 0
        self.difference_precision = difference_precision
        self.smallest_nodes = []
        self.smallest_length = np.inf
        self.k_smallest_lengths = np.zeros(k) + np.inf
        self.nodes = [initial_node]
        if depth:
            self.create_nodes(initial_node)
        
        #print('smallest length:',self.smallest_length)
        #print('smallest length points:',[n.index for n in self.smallest_nodes])
    
    def move_to_vector(self, move):
        return np.array({'A':[0,1],'B':[1,0], 'a': [0,-1], 'b': [-1,0]}[move])
    
    def move_to_matrix(self, move):
        return {'A': self.alpha1,'B': self.alpha2, 'a': mp.inverse(self.alpha1), 'b': mp.inverse(self.alpha2)}[move]

    def create_nodes(self, starting_node):
        
        if len(starting_node.index) == self.depth:
            return
        allowed_moves = ['A','B','a','b']
        for move_index in range(4):
            if not len(starting_node.index) or starting_node.index[-1] != allowed_moves[(move_index+2)%4]:
                next_node = Node(starting_node)
                next_node.index = f'{starting_node.index}{allowed_moves[move_index]}'
                next_node.coord = self.ratio**len(starting_node.index)*self.move_to_vector(allowed_moves[move_index]) + starting_node.coord
                next_node.matrix = starting_node.matrix*self.move_to_matrix(allowed_moves[move_index])
                next_node.length, _ =get_length(next_node.matrix)
                next_node.length =  np.float16(next_node.length)
                if round(next_node.length,1) > 0 and self.smallest_length > next_node.length + self.difference_precision:
                    self.smallest_length = min(next_node.length,self.smallest_length)
                    self.smallest_nodes = [next_node]
                elif round(next_node.length,1) > 0 and abs(self.smallest_length - next_node.length) < self.difference_precision:
                    self.smallest_length = min(self.smallest_length, next_node.length)
                    self.smallest_nodes.append(next_node)        

                self.k_smallest_lengths = k_smallest_lengths_add(self.k_smallest_lengths,next_node.length, self.difference_precision)


                self.nodes.append(next_node)
                self.create_nodes(next_node)


# #generate_sequence_layers(5)

# #alpha1 = np.array([[4,4,1],[2,3,1],[1,2,1]])
# #alpha2 = np.array([[0,0,1],[0,-1,-3],[1,6,9]])

# alpha1 = np.array([[3.5,3,0.5],[1.5,2,0.5],[1,2,1]])
# alpha2 = np.array([[0,0,1],[0,-0.5,-2.5],[2,7,10]])

# lengthheatmaptree = LengthHeatMapTree(7, 1/2, alpha1,alpha2)
# lengths = [node.length for node in lengthheatmaptree.nodes]
# # for node in lengthheatmaptree.nodes:
# #     print(node.length)
# fig = plt.figure()

# norm = mpl.colors.Normalize(vmin=0, vmax=max(lengths))
# cmap = cm.hot
# m = cm.ScalarMappable(norm=norm, cmap=cmap)
# # for node in lengthheatmaptree.nodes[1:]:
# #     plt.plot([node.coord[0],node.parent.coord[0]], [node.coord[1],node.parent.coord[1]], color='black')
# #     plt.scatter(node.coord[0],node.coord[1], color=m.to_rgba(node.length))
# ax = plt.axes(projection='3d')

# coordinates = []


# for node in lengthheatmaptree.nodes[1:]:
#     #ax.plot3D([node.coord[0],node.parent.coord[0]],  [node.coord[1],node.parent.coord[1]], [0,0], 'black')
#     coordinates.append([node.coord[0],node.coord[1],node.length])
#     #ax.scatter3D(node.coord[0],node.coord[1], node.length, s=0.5, color=m.to_rgba(node.length))

# coordinates = np.array(coordinates)
# X = coordinates[:,0]
# Y = coordinates[:,1]
# Z = coordinates[:,2]

# surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
# fig.colorbar(surf)

# ax.xaxis.set_major_locator(MaxNLocator(5))
# ax.yaxis.set_major_locator(MaxNLocator(6))
# ax.zaxis.set_major_locator(MaxNLocator(5))


# plt.show()
