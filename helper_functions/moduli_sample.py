import numpy as np
from sympy import linsolve, minimum
from add_new_triangle_functions import outitude_edge_params
from add_new_triangle_functions import compute_translation_matrix_torus
from length_heat_map import LengthHeatMapTree
import matplotlib.pyplot as plt

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
        #thetas = [1.69829298, 0.82466798, 0.08447248, 1.60624205, 2.46164864, 2.54761914, 1.24878935, 0.06283185]
        #thetas = [0.31001222, 0.4005024,  1.6318996,  1.58233146, 0.00833127, 1.71377308,3.13052548, 0.06283185]
        #thetas = np.pi*np.array([0.96261044, 0.39903471, 0.84415479, 0.27265972, 0.0644325,  0.31693688,0.1327247,  1.03426106])
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
        
        thetas = np.pi*np.array([0.31247444, 0.75, 0.99257343, 1, 0, 0,0.5, 0])
        
        for theta in theta_space:
            print(theta)
            thetas[3] = theta
            [radii, coordinates] = self.get_all_x_coordinates(thetas)
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
            #print(out_e, out_a,out_b)
            return False
    def get_single_x_coordinate(self,thetas,r):
        x = [np.cos(thetas[0])]
        for i in range(1,7):
            x.append(
                x[i-1]*np.tan(thetas[i-1])*np.cos(thetas[i])
            )
        x.append(x[-1]*np.sin(thetas[-1]))
        x = r*np.array(x)+1
        return x
        
    def get_all_x_coordinates(self,thetas):
        precision_halfs = 50
        number_of_halfs = 0
        original_h = 1
        h = original_h
        r = 0
        coordinates = []
        radii = []
        r_max = self.max_r
        while r < self.max_r:
            x = self.get_single_x_coordinate(thetas,r)

            #print(x)

            if not np.all([xi>0 for xi in x]):
                
                r_max = r-h
                break

            if not self.outitudes_positive(x):
                while not self.outitudes_positive(x):
                    r -= h
                    x = self.get_single_x_coordinate(thetas,r)
                
                h = h/2
                number_of_halfs += 1
                if precision_halfs-1 == number_of_halfs:
                    r_max = r
                    break
            r+=h

        radii = np.linspace(0,r_max,self.n)
        #print(radii)
        coordinates = np.array([self.get_single_x_coordinate(thetas,r) for r in radii])

        
        return [np.array(radii), np.array(coordinates)]
        







ModuliSample(100,10)