import numpy as np
from sympy import linsolve, minimum
from add_new_triangle_functions import outitude_edge_params
from add_new_triangle_functions import compute_translation_matrix_torus
from length_heat_map import LengthHeatMapTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ModuliSample():
    def __init__(self, max_r,n, tree_depth=2):
        self.n = n
        self.max_r = max_r
        self.tree_depth = tree_depth
        minimum_lengths_r_theta = []
        theta_n = 50
        theta_space = np.linspace(np.pi/theta_n,2*np.pi,theta_n)
        radiis = []
        thetas = np.pi*np.array([1,1,1,1,1,1,1,2])*np.random.random(8)
        print(thetas/np.pi)
        for i in range(len(thetas)-1):
            if thetas[i] == 0:
                thetas[i]+=0.05
            elif thetas[i] == np.pi:
                thetas[i]-=0.05
        
        if thetas[7] == 0:
            thetas[7] +=0.05
        elif thetas[7] == np.pi*2:
            thetas[7]-=0.05
        
        for theta in theta_space:
            print(theta)
            thetas[7] = theta
            [radii, coordinates] = self.get_x_coordinates(thetas)
            minimum_lengths = self.generate_minimum_length_distribution(coordinates)
            radiis.append(radii)
            minimum_lengths_r_theta.append(minimum_lengths)
        
        plt.figure()
        ax = plt.axes(projection='3d')
        for theta_index in range(len(theta_space)):
            radii = radiis[theta_index]
            theta = theta_space[theta_index]
            minimum_lengths = minimum_lengths_r_theta[theta_index]
            ax.plot3D(radii*np.cos(theta), radii*np.sin(theta), minimum_lengths)
        plt.show()



        
    
    def generate_minimum_length_distribution(self,coordinates):
        minimum_lengths = []
        for coordinate in coordinates:
            min_length = self.get_min_length_from_x(coordinate)
            minimum_lengths.append(min_length)
        return np.array(minimum_lengths)

    def get_min_length_from_x(self,x):
        alpha1,alpha2 = compute_translation_matrix_torus(x)
        lengthheatmaptree = LengthHeatMapTree(self.tree_depth, 1/2, alpha1,alpha2)
        min_length = lengthheatmaptree.smallest_length
        return min_length
    

    def outitudes_positive(self,x):
        [A,B,a_minus,a_plus,b_minus,b_plus,e_minus,e_plus] = x
        out_e = outitude_edge_params(A,B,a_minus,a_plus,b_minus,b_plus,e_minus,e_plus)
        out_a = outitude_edge_params(A,B,b_minus,b_plus, e_minus, e_plus, a_minus, a_plus)
        out_b = outitude_edge_params(A,B,e_minus, e_plus,a_minus, a_plus, b_minus, b_plus)
        if out_e >= 0 and out_a >=0 and out_b >= 0:
            return True
        else:
            return False
        
    def get_x_coordinates(self,thetas):
        
        r_space = list(np.linspace(0,self.max_r,self.n)[::-1])
        r = 0
        coordinates = []
        radii = []
        

        while r < self.max_r:
            r = r_space.pop()
            x = [np.cos(thetas[0])]
            for i in range(1,7):
                x.append(
                    x[i-1]*np.tan(thetas[i-1])*np.cos(thetas[i])
                )
            x.append(x[-1]*np.sin(thetas[-1]))
            x = r*np.array(x)+1
            if not np.all([xi>0 for xi in x]) or not self.outitudes_positive(x):
                break
            
            coordinates.append(x)
            radii.append(r)
        return [np.array(radii), np.array(coordinates)]
        







ModuliSample(100,100)