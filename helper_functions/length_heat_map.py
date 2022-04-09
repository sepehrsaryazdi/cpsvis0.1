import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self,parent=None):
        self.parent = parent


class LengthHeatMapTree:
    def __init__(self,depth, ratio=1/2):
        self.depth = depth
        self.ratio = ratio
        initial_node = Node()
        initial_node.index = ''
        initial_node.coord = np.array([0,0])
        self.nodes = [initial_node]
        if depth:
            self.create_nodes(initial_node)
    
    def move_to_vector(self, move):
        return np.array({'R':[0,1],'U':[1,0], 'L': [0,-1], 'D': [-1,0]}[move])

    def create_nodes(self, starting_node):
        if len(starting_node.index) == self.depth:
            return
        allowed_moves = ['R','U','L','D']
        for move_index in range(4):
            if not len(starting_node.index) or starting_node.index[-1] != allowed_moves[(move_index+2)%4]:
                next_node = Node(starting_node)
                next_node.index = f'{starting_node.index}{allowed_moves[move_index]}'
                next_node.coord = self.ratio**len(starting_node.index)*self.move_to_vector(allowed_moves[move_index]) + starting_node.coord 
                self.nodes.append(next_node)
                self.create_nodes(next_node)
                
        






def generate_sequence_layers(depth):

    # depth = max length of sequence generated

    
    if depth == 0:
        return [[[[0,0]]]]
    
    allowed_moves = [[1,1],[2,1],[1,-1],[2,-1]]

    layers = [[[[0,0]]],[[[0,0],allowed_moves[0]],[[0,0],allowed_moves[1]],[[0,0],allowed_moves[2]],[[0,0],allowed_moves[3]]]]
    
    while len(layers)-1 <= depth:
        previous_layer = layers[-1]
        next_layer = []
        for sequence in previous_layer:
            for next_move_index in range(len(allowed_moves)):
                if sequence[-1] != allowed_moves[(next_move_index+2)%len(allowed_moves)]:
                    
                    next_sequence = sequence.copy()
                    next_sequence.append(allowed_moves[next_move_index])
                    next_layer.append(next_sequence)
        layers.append(next_layer)    
    for layer in layers:
        print(len(layer))

def length_heat_map(layers):
    for layer in layers:
        pass




    pass


#generate_sequence_layers(5)

lengthheatmaptree = LengthHeatMapTree(7, 1/2)
plt.figure()
for node in lengthheatmaptree.nodes[1:]:
    plt.plot([node.coord[0],node.parent.coord[0]], [node.coord[1],node.parent.coord[1]], color='blue')
plt.show()
