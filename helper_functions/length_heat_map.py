import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from helper_functions.add_new_triangle_functions import *


class Node:
    def __init__(self,parent=None):
        self.parent = parent


class LengthHeatMapTree:
    def __init__(self,depth, ratio=1/2, alpha1=np.identity(3), alpha2=np.identity(3)):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.depth = depth
        self.ratio = ratio
        initial_node = Node()
        initial_node.index = ''
        initial_node.coord = np.array([0,0])
        initial_node.matrix = np.identity(3)
        initial_node.length = 0
        self.nodes = [initial_node]
        if depth:
            self.create_nodes(initial_node)
    
    def move_to_vector(self, move):
        return np.array({'R':[0,1],'U':[1,0], 'L': [0,-1], 'D': [-1,0]}[move])
    
    def move_to_matrix(self, move):
        return {'R': self.alpha1,'U': self.alpha2, 'L': np.linalg.inv(self.alpha1), 'D': np.linalg.inv(self.alpha2)}[move]

    def create_nodes(self, starting_node):
        if len(starting_node.index) == self.depth:
            return
        allowed_moves = ['R','U','L','D']
        for move_index in range(4):
            if not len(starting_node.index) or starting_node.index[-1] != allowed_moves[(move_index+2)%4]:
                next_node = Node(starting_node)
                next_node.index = f'{starting_node.index}{allowed_moves[move_index]}'
                next_node.coord = self.ratio**len(starting_node.index)*self.move_to_vector(allowed_moves[move_index]) + starting_node.coord
                next_node.matrix = np.matmul(starting_node.matrix,self.move_to_matrix(allowed_moves[move_index]))
                next_node.length, _ = get_length(next_node.matrix)
                self.nodes.append(next_node)
                self.create_nodes(next_node)


# #generate_sequence_layers(5)

# #alpha1 = np.array([[4,4,1],[2,3,1],[1,2,1]])
# #alpha2 = np.array([[0,0,1],[0,-1,-3],[1,6,9]])

# alpha1 = np.array([[3.5,3,0.5],[1.5,2,0.5],[1,2,1]])
# alpha2 = np.array([[0,0,1],[0,-0.5,-2.5],[2,7,10]])

# lengthheatmaptree = LengthHeatMapTree(6, 1/2, alpha1,alpha2)
# lengths = [node.length for node in lengthheatmaptree.nodes]
# for node in lengthheatmaptree.nodes:
#     print(node.length)
# plt.figure()

# norm = mpl.colors.Normalize(vmin=0, vmax=max(lengths))
# cmap = cm.hot
# m = cm.ScalarMappable(norm=norm, cmap=cmap)
# for node in lengthheatmaptree.nodes[1:]:
#     plt.plot([node.coord[0],node.parent.coord[0]], [node.coord[1],node.parent.coord[1]], color='black')
#     plt.scatter(node.coord[0],node.coord[1], color=m.to_rgba(node.length))
# #ax = plt.axes(projection='3d')
# # for node in lengthheatmaptree.nodes[1:]:
# #     ax.plot3D([node.coord[0],node.parent.coord[0]],  [node.coord[1],node.parent.coord[1]], [0,0], 'black')
# #     ax.scatter3D(node.coord[0],node.coord[1], node.length, s=0.5)
# plt.show()
