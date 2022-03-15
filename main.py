from enum import unique
from fnmatch import translate
from math import gamma
import tkinter as tk
from tkinter import ttk
from numpy import arctan2

from triangle_class.abstract_triangle import AbstractSurface, AbstractVertex, AbstractEdge
from triangle_class.decorated_triangle import *
from visualise.surface_vis import SurfaceVisual
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helper_functions.add_new_triangle_functions import *
from tkinter import filedialog
import pandas as pd
import pickle
import sys
import os


def clover_position(x, t):
    x = np.array(x)
    x = x/sum(sum(x))

    T_R2e = [[-1/np.sqrt(2), 1/np.sqrt(2), 0],[-1/np.sqrt(6), -1/np.sqrt(6), np.sqrt(2/3)],[1,1,1]]

    cube_root1 = np.array([1,0])
    cube_root2 = np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    cube_root3 = np.array([np.cos(-2*np.pi/3), np.sin(-2*np.pi/3)])

    alpha_10 = np.sqrt(2)*(cube_root2-cube_root1)/2
    alpha_01 = -np.sqrt(6)/2*(cube_root1 + cube_root2)

    [x,y,z] = np.matmul(T_R2e,x)

    [x,y] = [x[0],y[0]]
    [x, y] = [x / z, y / z]

    [x,y] = x*alpha_10 + y*alpha_01

    return [x,y]



class TranslationLength:
    def __init__(self):
        self.tk = tk
        self.win = self.tk.Toplevel()
        self.win.wm_title("Translation Lengths and Length Spectrum")
        self.l = tk.Label(self.win, text="Enter a fundamental group product string using the format below.")
        self.l.pack(padx=20, pady=10)
        self.abstract_surface = app.abstract_surface
        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_type = FigureCanvasTkAgg(self.figure, self.win)
        self.generate_combinatorial_map()
        self.chart_type.get_tk_widget().pack()
        self.instructions = tk.Label(self.win, text="Referencing to an index of available generators shown above, write in the text box below the index and the power desired.\nClick \"Add String\" to multiply the string by the new generator product on the right. The string is interpreted as being ordered from left to right.")
        self.instructions.pack(padx=20, pady=10)
        self.product_string_frame = ttk.Frame(self.win)
        self.gamma_equals_label = tk.Label(self.product_string_frame,text="γ = ",font=("Courier", 30))
        self.product_string = tk.StringVar()
        self.product_string_label = tk.Label(self.product_string_frame, textvariable=self.product_string, fg="blue",font=("Courier", 30))

        self.product_string_data = [[1,2]]

        self.product_string.set("(α₁)²")
        self.gamma_equals_label.pack(side="left")
        self.product_string_label.pack(side="left", pady=(0,5))
        self.product_string_frame.pack(side="top")
        self.enter_string_frame = ttk.Frame(self.win)
        self.new_product_text = tk.Label(self.enter_string_frame, text="Next String Term: ")
        self.new_product_text.pack(side="left")
        self.new_product_string_label = tk.Label(self.enter_string_frame, text="α", fg="blue",font=("Courier", 44))
        self.new_product_string_label.pack(side="left")
        self.string_entries_frame = ttk.Frame(self.enter_string_frame)
        self.enter_string_power_string = tk.StringVar(value="1")
        self.enter_string_power_entry = ttk.Entry(self.string_entries_frame,textvariable=self.enter_string_power_string, width=3)
        self.enter_string_index_string = tk.StringVar(value="1")
        self.enter_string_index_entry = ttk.Entry(self.string_entries_frame,textvariable=self.enter_string_index_string,  width=3)
        self.enter_string_power_entry.pack(side="top")
        self.enter_string_index_entry.pack(side="top")
        self.string_entries_frame.pack(side="left")
        self.add_string_button = ttk.Button(self.enter_string_frame, text="Add String")
        self.add_string_button.pack(side="left", padx=25)
        self.enter_string_frame.pack()
        self.clear_string_button = ttk.Button(self.win,text="Clear String")
        self.compute_translation_length_button = ttk.Button(self.win, text="Compute Translation Length")
        self.clear_string_button.pack(side="left", anchor="nw", padx=25,pady=25)
        self.error_message_string = tk.StringVar(value="")
        self.error_message = tk.Label(self.win,textvariable=self.error_message_string, fg="red")
        self.error_message.pack(side="left",padx=5,pady=25)
        self.compute_translation_length_button.pack(side="right",anchor="ne",padx=25,pady=25)
        self.compute_translation_matrices()
        self.add_string_button.bind("<ButtonPress>", self.add_string)
        self.clear_string_button.bind("<ButtonPress>", self.clear_string)
        self.compute_translation_length_button.bind("<ButtonPress>", self.compute_translation_length)


    def compute_matrix_path(self, edge_starting, initial_triangle, final_triangle, final_edge):
        triangle_list = np.array(self.abstract_plotting_surface.triangles)
        final_triangle_list_index = np.where(triangle_list == final_triangle)[0][0]
        current_triangle = initial_triangle
        current_triangle_list_index = np.where(triangle_list == current_triangle)[0][0]
        product = np.identity(3)
        product_terms = ['identity']
        intersecting_edge = None
        current_edge = edge_starting
        while current_triangle != final_triangle:
            
            if final_triangle_list_index < current_triangle_list_index:            
                    
                for edge in current_triangle.edges:
                    other_edge = edge.edge_glued[2]
                    if other_edge.triangle == triangle_list[current_triangle_list_index-1] and ((np.all(edge.v0.coord == other_edge.v0.coord) and np.all(edge.v1.coord == other_edge.v1.coord)) or (np.all(edge.v0.coord == other_edge.v1.coord) and np.all(edge.v1.coord == other_edge.v0.coord))):
                        intersecting_edge = edge

                current_edge_index = 0
                intersecting_edge_index = 0
                for index in range(3):
                    if current_edge == current_edge.triangle.edges[index]:
                        current_edge_index = index
                    if intersecting_edge == current_edge.triangle.edges[index]:
                        intersecting_edge_index = index
                
                if (current_edge_index+1) % 3 == intersecting_edge_index:
                    product = np.matmul(triangle_matrix(current_triangle.triangle_parameter),product)
                    product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, not inverted")
                elif (current_edge_index-1) % 3 == intersecting_edge_index:
                    product = np.matmul(np.linalg.inv(triangle_matrix(current_triangle.triangle_parameter)),product)
                    product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, inverted")

                    
                if intersecting_edge.index != '02':
                    product = np.matmul(edge_matrix(intersecting_edge.ea, intersecting_edge.eb), product)
                    product_terms.append(f"{intersecting_edge.index, intersecting_edge.triangle.index, intersecting_edge.ea, intersecting_edge.eb}")
                else:
                    product = np.matmul(edge_matrix(intersecting_edge.eb, intersecting_edge.ea), product)
                    product_terms.append(f"{intersecting_edge.index, intersecting_edge.triangle.index, intersecting_edge.eb, intersecting_edge.ea}")
                
                current_triangle = triangle_list[current_triangle_list_index-1]
                current_triangle_list_index = np.where(triangle_list == current_triangle)[0][0]
                for edge in current_triangle.edges:
                    if edge.index == intersecting_edge.edge_glued[2].index:
                        current_edge = edge

            else:
                
                for edge in current_triangle.edges:
                    other_edge = edge.edge_glued[2]
                    if other_edge.triangle == triangle_list[current_triangle_list_index+1] and ((np.all(edge.v0.coord == other_edge.v0.coord) and np.all(edge.v1.coord == other_edge.v1.coord)) or (np.all(edge.v0.coord == other_edge.v1.coord) and np.all(edge.v1.coord == other_edge.v0.coord))):
                        intersecting_edge = edge

                current_edge_index = 0
                intersecting_edge_index = 0
                for index in range(3):
                    if current_edge == current_edge.triangle.edges[index]:
                        current_edge_index = index
                    if intersecting_edge == current_edge.triangle.edges[index]:
                        intersecting_edge_index = index
                
                if (current_edge_index+1) % 3 == intersecting_edge_index:
                    product = np.matmul(triangle_matrix(current_triangle.triangle_parameter),product)
                    product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, not inverted")
                elif (current_edge_index-1) % 3 == intersecting_edge_index:
                    product = np.matmul(np.linalg.inv(triangle_matrix(current_triangle.triangle_parameter)),product)
                    product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, inverted")
                
                if intersecting_edge.index != '02':
                    product = np.matmul(edge_matrix(intersecting_edge.ea, intersecting_edge.eb), product)
                    product_terms.append(f"{intersecting_edge.index, intersecting_edge.triangle.index, intersecting_edge.ea, intersecting_edge.eb}")
                else:
                    product = np.matmul(edge_matrix(intersecting_edge.eb, intersecting_edge.ea), product)
                    product_terms.append(f"{intersecting_edge.index, intersecting_edge.triangle.index, intersecting_edge.eb, intersecting_edge.ea}")
                
                current_triangle = triangle_list[current_triangle_list_index+1]
                current_triangle_list_index = np.where(triangle_list == current_triangle)[0][0]
                for edge in current_triangle.edges:
                    if edge.index == intersecting_edge.edge_glued[2].index:
                        current_edge = edge
        
        if current_edge != final_edge:
            for index in range(3):
                edge = current_triangle.edges[index]
                if edge == current_edge:
                    current_edge_index = index
                if edge == final_edge:
                    final_edge_index = index
            if (current_edge_index + 1)%3 == final_edge_index:
                product = np.matmul(triangle_matrix(current_triangle.triangle_parameter),product)
                product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, not inverted")
            elif (current_edge_index-1) % 3 == final_edge_index:
                product = np.matmul(np.linalg.inv(triangle_matrix(current_triangle.triangle_parameter)),product)
                product_terms.append(f"{current_triangle.index, current_triangle.triangle_parameter}, inverted")
        #print(product_terms)
        return (product, final_edge)
    
    
    def compute_translation_matrices(self):
        middle_index = int(np.median(np.linspace(1,len(self.abstract_plotting_surface.triangles))))-1
        triangle_list = self.abstract_plotting_surface.triangles
        centre_triangle = triangle_list[middle_index]
        
        self.representations = []
        
        for edge in self.boundary_edges:
            edge_to_reach = edge.edge_glued[2]
            
            central_edge = centre_triangle.edges[1]

            first_matrix, _  = self.compute_matrix_path(central_edge, centre_triangle,edge_to_reach.triangle, edge_to_reach)
            
            if edge.index != '02':
                final_edge_matrix = edge_matrix(edge.eb, edge.ea)
                #print('edges params: ',edge.eb,edge.ea)
            else:
                #print('edges params: ',edge.eb,edge.ea)
                final_edge_matrix = edge_matrix(edge.ea, edge.eb)

            second_matrix, _ = self.compute_matrix_path(edge, edge.triangle, centre_triangle, central_edge)
            
            

            product = second_matrix @ final_edge_matrix @ first_matrix
            self.representations.append(product)

            #print(np.linalg.det(product))
            #print(product)

            #t_matrix = triangle_matrix(1)
            #e_matrix = edge_matrix(1,1)
            #first_expected = t_matrix @ e_matrix @ np.linalg.inv(t_matrix) @ e_matrix @ t_matrix @ e_matrix
            #second_expected = np.linalg.inv(t_matrix) @ e_matrix @ np.linalg.inv(t_matrix) @ e_matrix @ t_matrix
            #print(second_expected @ e_matrix @ first_expected)
            #print(np.linalg.inv(t_matrix) @ e_matrix @ np.linalg.inv(t_matrix) @ e_matrix @ t_matrix @ e_matrix @ np.linalg.inv(t_matrix) @ e_matrix @ t_matrix @ e_matrix)
            #print(product)
            
            
            #product = np.matmul(final_edge_matrix,np.matmul(first_matrix,second_matrix))
            #rint(np.sort(np.absolute(np.linalg.eigvals(self.representations[-1]))), np.sort(np.absolute(np.linalg.eigvals(product))))

     

    def compute_translation_length(self, event):
        
        product = np.identity(3)
        for product_data in self.product_string_data[::-1]:
            [index, power] = product_data
            if power != 0:
                representation = self.representations[index-1]
                product = np.matmul(np.linalg.matrix_power(representation, power),product)
        
        eigenvalues = np.linalg.eigvals(product)
        absolute_eigenvalues = np.absolute(eigenvalues)
        absolute_eigenvalues = np.sort(absolute_eigenvalues)
        smallest_eigenvalue = absolute_eigenvalues[0]
        largest_eigenvalue = absolute_eigenvalues[-1]

        length = np.log(largest_eigenvalue/smallest_eigenvalue)
        #print(length)
        eigenvalues_string = f"{absolute_eigenvalues}"[1:-1].split(' ')
        eigenvalues_string = ', '.join(eigenvalues_string)
        self.error_message_string.set(f"Length: {length}\nModulus of Eigenvalues: {eigenvalues_string}.")

        


    def clear_string(self, event):
        self.product_string_data = []
        total_string = ["𝟙"]
        self.product_string.set("".join(total_string))
        

    def add_string(self,event):
        boundary_list = [i+1 for i in range(len(self.boundary_edges))]
        try:
            assert int(self.enter_string_power_string.get()) == app.string_fraction_to_float(self.enter_string_power_string.get())
            self.error_message_string.set("")
        except:
            self.error_message_string.set("Please ensure that the power is an integer before adding string.")
            return
        try:
            assert int(self.enter_string_index_string.get()) and int(self.enter_string_index_string.get()) in boundary_list
            self.error_message_string.set("")
        except:
            
            boundary_string = f"{boundary_list}"[1:-1]
            self.error_message_string.set(f"Please ensure that the index is valid before adding string.\n The valid indices are: {boundary_string}.")
            return
        
        added_element = [int(self.enter_string_index_string.get()), int(self.enter_string_power_string.get())]
        if len(self.product_string_data) != 0:
            last_element = self.product_string_data[-1]
            if last_element[0] == added_element[0]:
                last_element[1]+=added_element[1]
            else:
                self.product_string_data.append(added_element)
            if self.product_string_data[-1][1] == 0:
                self.product_string_data.pop()
        else:
            self.product_string_data.append(added_element)
        if len(self.product_string_data) == 0:
            total_string = ["𝟙"]
        else:
            total_string = []
            for data in self.product_string_data:
                if data[1] == 1:
                    total_string.append(f"(α{integer_to_script(data[0],False)})")
                else:
                    total_string.append(f"(α{integer_to_script(data[0],False)}){integer_to_script(data[1],True)}")
        
        self.product_string.set("".join(total_string))

        
    
    def find_last_vertex(self,vertex, glued_edge_belonging_to):
        count = 2
        while count == 2:
            vertex = self.get_dual_vertex(vertex, glued_edge_belonging_to)
            for edge in vertex.edges:
                if edge != glued_edge_belonging_to and edge.edge_glued:
                    glued_edge_belonging_to = edge
                    break
            count = 0
            for edge in vertex.edges:
                if edge.edge_glued:
                    count += 1
        return vertex, glued_edge_belonging_to

    def give_edge_identification_color_and_arrow(self):

        colors_ = lambda n: list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF), range(n)))


        edges = []
        for plotting_triangle in self.abstract_plotting_surface.triangles:
            abstract_triangle = self.abstract_surface.triangles[plotting_triangle.index]
            plotting_triangle.triangle_parameter = abstract_triangle.triangle_parameter
            for abstract_edge in abstract_triangle.edges:
                try:
                    abstract_edge.edge_glued[2]
                    flipped = (abstract_edge.edge_glued[1] != abstract_edge.edge_glued[2].v0)
                    edge_to_glue = None
                    for edge in plotting_triangle.edges:
                        if edge.index == abstract_edge.index:
                            edge_to_glue = edge
                            edge_to_glue.ea = abstract_edge.ea
                            edge_to_glue.eb = abstract_edge.eb

                    abstract_other_edge = abstract_edge.edge_glued[2]
                    other_edge_to_glue = None
                    for other_triangle in self.abstract_plotting_surface.triangles:
                        for edge in other_triangle.edges:
                            if edge.index == abstract_other_edge.index and other_triangle.index == abstract_other_edge.triangle.index:
                                other_edge_to_glue = edge
                                other_edge_to_glue.ea = abstract_other_edge.ea
                                other_edge_to_glue.eb = abstract_other_edge.eb
                    if not flipped:
                        self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0,
                                                             other_edge_to_glue.v0)
                    else:
                        self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0,
                                                             other_edge_to_glue.v1)
                except:
                    pass

        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                edges.append(edge)

        colors = colors_(2*len(edges)+1)
        arrow_strokes = 1
        for edge in edges:
            new_color = colors.pop()
            edge.color = new_color
            edge.arrow_strokes = arrow_strokes
            try:
                edge.edge_glued[2].color = new_color
                edge.edge_glued[2].arrow_strokes = arrow_strokes
                arrow_strokes += 1
            except:
                edge.color = colors.pop()
                edge.arrow_strokes = 0
        min_stroke = len(edges)+1
        for edge in edges:
            if edge.arrow_strokes > 0:
                min_stroke = min(min_stroke, edge.arrow_strokes)
        for edge in edges:
            if edge.arrow_strokes > 0:
                edge.arrow_strokes = edge.arrow_strokes - min_stroke + 1
    
    def glue_plotting_surface_edges(self):
        for index in range(len(self.abstract_plotting_surface.triangles[:-1])):
           
            triangle_plotting_index = self.abstract_plotting_surface.triangles[index].index
            self.abstract_plotting_surface.triangles[index].triangle_parameter = self.abstract_surface.triangles[triangle_plotting_index].triangle_parameter
            next_triangle_plotting_index = self.abstract_plotting_surface.triangles[index+1].index
            edge_connection_index = '01'
            other_edge_index = '01'
            flipped = 0
            for edge in self.abstract_surface.triangles[triangle_plotting_index].edges:
                try:
                    if edge.edge_glued[2].triangle == self.abstract_surface.triangles[next_triangle_plotting_index]:
                        edge_connection_index = edge.index
                        other_edge_index = edge.edge_glued[2].index
                        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                except:
                    pass
            edge_to_glue = self.abstract_plotting_surface.triangles[index].edges[0]
            other_edge_to_glue = self.abstract_surface.triangles[next_triangle_plotting_index].edges[0]
            for edge in self.abstract_plotting_surface.triangles[index].edges:
                if edge.index == edge_connection_index:
                    edge_to_glue = edge
            for edge in self.abstract_plotting_surface.triangles[index+1].edges:
                if edge.index == other_edge_index:
                    other_edge_to_glue = edge
            if not flipped:
                self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v0)
            else:
                self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v1)
    
    def get_dual_vertex(self,vertex, edge):
        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
        vertex_is_at_end_of_edge = (edge.v1 == vertex)
        if not flipped:
            if vertex_is_at_end_of_edge:
                other_vertex = edge.edge_glued[2].v1
            else:
                other_vertex = edge.edge_glued[2].v0
        else:
            if vertex_is_at_end_of_edge:
                other_vertex = edge.edge_glued[2].v0
            else:
                other_vertex = edge.edge_glued[2].v1
        return other_vertex

    def triangle_order_generator(self,edge_list, prev_state, n, top_bottom_list):
        if len(edge_list) == n:
            return (edge_list,top_bottom_list)
        current_edge_on_previous_triangle = edge_list[-1]
        current_edge_on_new_triangle = current_edge_on_previous_triangle.edge_glued[2]
        new_triangle = current_edge_on_new_triangle.triangle
        flipped = 0
        if current_edge_on_previous_triangle.edge_glued[1] != current_edge_on_new_triangle.v0:
            flipped = 1
        for edge_index in range(3):
            if new_triangle.edges[edge_index] == current_edge_on_new_triangle:
                current_edge_on_new_triangle = edge_index
                break
        new_prev_state = 'top'
        if prev_state == 'top':
            next_edge_on_new_triangle = (current_edge_on_new_triangle + ((-1)**flipped)*1) % 3
            if next_edge_on_new_triangle != (current_edge_on_new_triangle + 1) % 3:
                new_prev_state = 'bottom'
        else:
            next_edge_on_new_triangle = (current_edge_on_new_triangle - ((-1) ** flipped) * 1) % 3
            if next_edge_on_new_triangle != (current_edge_on_new_triangle + 1) % 3:
                new_prev_state = 'bottom'

        top_bottom_list.append(new_prev_state)

        next_edge_on_new_triangle = new_triangle.edges[next_edge_on_new_triangle]
        edge_list.append(next_edge_on_new_triangle)
        return self.triangle_order_generator(edge_list, new_prev_state, n, top_bottom_list)
    
    def vertex_traversal(self,vertex, vertex_points):

        try:
            self.abstract_plotting_surface.give_vertex_coordinates(vertex,vertex_points.pop())
            self.vertex_traversed_list.append(vertex)
        except:
            return

        count = 0
        for edge in vertex.edges:
            if edge.edge_glued:
                count+=1

        triangle_belongs_to = vertex.edges[0].triangle
        if count == 0:
            for other_vertex_index in [(vertex.index-1)%3,(vertex.index+1)%3]:
                coord = triangle_belongs_to.vertices[other_vertex_index].coord
                if not len(coord):
                    vertex = triangle_belongs_to.vertices[other_vertex_index]
                    break
        elif count == 1:

            other_vertex_to_consider = None
            non_glued_edge = None
            for edge in vertex.edges:
                if not edge.edge_glued:
                    non_glued_edge = edge
            for other_vertex in [non_glued_edge.v0, non_glued_edge.v1]:
                if other_vertex != vertex:
                    other_vertex_to_consider = other_vertex
            if not len(other_vertex_to_consider.coord):
                vertex = other_vertex_to_consider
            else:
                glued_edge_belonging_to = None
                for edge in vertex.edges:
                    if edge.edge_glued:
                        glued_edge_belonging_to = edge
                other_vertex_to_consider = self.get_dual_vertex(vertex, glued_edge_belonging_to)
                other_count = 0
                for edge in other_vertex_to_consider.edges:
                    if edge.edge_glued:
                        other_count+=1

                if other_count == 1:
                    for other_other_vertex_index in [(other_vertex_to_consider.index - 1) % 3, (other_vertex_to_consider.index + 1) % 3]:
                        coord = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index].coord
                        if not len(coord):
                            vertex = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index]
                            break
                else:
                    other_vertex_to_consider, glued_edge_belonging_to = self.find_last_vertex(vertex, glued_edge_belonging_to)
                    for other_other_vertex_index in [(other_vertex_to_consider.index - 1) % 3, (other_vertex_to_consider.index + 1) % 3]:
                        coord = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index].coord
                        if not len(coord):
                            vertex = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index]
                            break
        return self.vertex_traversal(vertex, vertex_points)

    def plot_combinatorial_map(self):
        self.ax.clear()
        self.ax.remove()
        self.ax.set_axis_off()

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Fundamental Group Generators')
        self.ax.set_axis_off()
        #self.figure.canvas.mpl_connect('button_press_event',self.select_triangle_combinatorial_map)
        plotted_edges = []
        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                [x1, y1] = edge.v0.coord
                [x2, y2] = edge.v1.coord
                x = [x1, x2]
                y = [y1, y2]
                self.ax.plot(x, y, c=edge.color)
                plotted_edges.append(edge)
                if edge.arrow_strokes > 0:
                    try:
                        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                        for i in range(edge.arrow_strokes):
                            if flipped and edge.edge_glued[2] in plotted_edges:
                                [x1, y1] = edge.v1.coord
                                [x2, y2] = edge.v0.coord
                            self.ax.arrow(x1, y1, (i + 4) * (x2 - x1) / (edge.arrow_strokes + 7),
                                     (i + 4) * (y2 - y1) / (edge.arrow_strokes + 7), head_width=0.3, color=edge.color)
                    except:
                        pass

        for triangle in self.abstract_plotting_surface.triangles:
            [x1, y1] = triangle.vertices[0].coord
            [x2, y2] = triangle.vertices[1].coord
            [x3, y3] = triangle.vertices[2].coord
            x = [x1, x2, x3, x1]
            y = [y1, y2, y3, y1]
            if triangle.selected:
                self.ax.fill(x,y, "b", alpha=0.2)
            self.ax.annotate(triangle.index, [np.mean(x[:-1]), np.mean(y[:-1])])
            coord0 = np.array([x1, y1])
            coord1 = np.array([x2, y2])
            coord2 = np.array([x3, y3])
            self.ax.annotate(0, 9 * coord0 / 10 + 1 / 10 * (coord1 + coord2), color='grey')
            self.ax.annotate(1, 9 * coord1 / 10 + 1 / 10 * (coord0 + coord2), color='grey')
            self.ax.annotate(2, 9 * coord2 / 10 + 1 / 10 * (coord1 + coord0), color='grey')

        i=1
        for edge in self.boundary_edges:
            [x1,y1] = edge.v0.coord
            [x2,y2] = edge.v1.coord
            [x3,y3] = edge.edge_glued[2].v0.coord
            [x4,y4] = edge.edge_glued[2].v1.coord
            e_midpoint = np.array([1/2*(x1+x2),1/2*(y1+y2)])
            e_glued_midpoint = np.array([1/2*(x3+x4),1/2*(y3+y4)])
            curve_function = beziercurve(e_midpoint, np.array([0,0]), e_glued_midpoint)
            t_vals = np.linspace(0.03,0.97,50)
            gamma_coordinates = []
            for t in t_vals:
                gamma_coordinates.append(curve_function(t))
            gamma_coordinates = np.array(gamma_coordinates)
            self.ax.plot(gamma_coordinates[:,0],gamma_coordinates[:,1], c=edge.color)
            self.ax.arrow(gamma_coordinates[25,0], gamma_coordinates[25,1], gamma_coordinates[26,0]-gamma_coordinates[25,0], gamma_coordinates[26,1]-gamma_coordinates[25,1],head_width=0.3, color=edge.color)
            self.ax.arrow(gamma_coordinates[10,0], gamma_coordinates[10,1], gamma_coordinates[11,0]-gamma_coordinates[10,0], gamma_coordinates[11,1]-gamma_coordinates[10,1],head_width=0.3, color=edge.color)
            self.ax.arrow(gamma_coordinates[40,0], gamma_coordinates[40,1], gamma_coordinates[41,0]-gamma_coordinates[40,0], gamma_coordinates[41,1]-gamma_coordinates[40,1],head_width=0.3, color=edge.color)
            text_position = 1/2*(gamma_coordinates[40,:]+gamma_coordinates[30,:])
            text_position+=1/2*np.array([-2*np.sign(text_position[0]),1])
            self.ax.annotate(rf'$\alpha_{i}$',text_position,color=edge.color)
            i+=1

        self.chart_type.draw()

    def generate_combinatorial_map(self):
        first_triangle = self.abstract_surface.triangles[0]
        first_edge_index = 0
        for edge_index in range(3):
            if not first_triangle.edges[edge_index].edge_glued:
                first_edge_index = (edge_index - 1) % 3
                break
        edge_list = [first_triangle.edges[first_edge_index]]
        edge_list, top_bottom_list = self.triangle_order_generator(edge_list, 'top', len(self.abstract_surface.triangles), ['top'])

        triangle_indices = [edge.triangle.index for edge in edge_list]


        vertex_points = []
        a = 10
        b = 10
        thetas = np.linspace(2/(len(edge_list))*np.pi,2*np.pi+1/(len(edge_list))*np.pi, len(edge_list)+2)
        for theta in thetas:
            vertex_points.append(np.array([a*np.cos(theta),-b*np.sin(theta)]))

        self.abstract_plotting_surface = AbstractSurface()
        for triangle_index in triangle_indices:
            self.abstract_plotting_surface.add_triangle()
            self.abstract_plotting_surface.triangles[-1].index = triangle_index

        self.glue_plotting_surface_edges()
        
        self.vertex_traversed_list = []
        self.vertex_traversal(self.abstract_plotting_surface.triangles[0].vertices[0], vertex_points)
        orientation_first_triangle = np.linalg.det([[v.coord[0], v.coord[1], 1] for v in self.abstract_plotting_surface.triangles[0].vertices])
        if orientation_first_triangle < 0:
            unique_vertices = []
            for triangle in self.abstract_plotting_surface.triangles:
                for vertex in triangle.vertices:
                    if vertex not in unique_vertices:
                        unique_vertices.append(vertex)
            for vertex in unique_vertices:
                vertex.coord = vertex.coord[::-1]
            self.abstract_plotting_surface.triangles = self.abstract_plotting_surface.triangles[::-1]


        vertex_angles = []
        for vertex in self.vertex_traversed_list:
            vertex_angles.append(arctan2(vertex.coord[1],vertex.coord[0]))
        
        self.vertex_traversed_list = np.array(self.vertex_traversed_list)[np.argsort(vertex_angles)]

        

        self.boundary_edges = []
        for i in range(len(self.vertex_traversed_list)):
            self.boundary_edges.append(0)
        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                for vertex_index in range(len(self.vertex_traversed_list)):
                    vertex = self.vertex_traversed_list[vertex_index]
                    next_vertex = self.vertex_traversed_list[(vertex_index+1)%len(self.vertex_traversed_list)]
                    if (np.all(edge.v0.coord == vertex.coord) and np.all(edge.v1.coord == next_vertex.coord)) or (np.all(edge.v1.coord == vertex.coord) and np.all(edge.v0.coord == next_vertex.coord)):
                        self.boundary_edges[vertex_index] = edge
        
       
        


        self.give_edge_identification_color_and_arrow()     

        unique_boundaries = []
        for boundry_edge in self.boundary_edges:
            if boundry_edge.edge_glued[2] not in unique_boundaries:
                unique_boundaries.append(boundry_edge)

        self.boundary_edges = unique_boundaries
    


        self.plot_combinatorial_map()


        


class MSL3R:
    def __init__(self):
        self.tk = tk
        self.parameter_entries = {}
        self.triangle_parameter_entry = None
        self.win = self.tk.Toplevel()
        self.win.resizable(width=False, height=False)
        self.win.wm_title("Enter M from SL(3,R)")
        self.l = tk.Label(self.win,
                           text="Enter the matrix M from SL(3,ℝ) below to apply on the decorations (R • M⁻¹, M • C).")
        self.l.pack(padx=20, pady=10)


        self.m_equals_text_frame = ttk.Frame(self.win)
        self.m_equals_text = tk.Label(self.m_equals_text_frame, text="M = ")
        self.m_equals_text.pack(side="left",padx=0,pady=5)



        self.matrix_frame = ttk.Frame(self.m_equals_text_frame)

        self.matrix_entries = []
        self.row_frames = []
        for i in range(3):
            self.row_frames.append(ttk.Frame(self.matrix_frame))

        j=0
        for i in range(9):
            self.matrix_entries.append(ttk.Entry(self.row_frames[j], width=8))
            if (i+1) % 3 == 0:
                j+=1



        for j in range(3):
            for entry in self.matrix_entries[3*j:3*j+3]:
                entry.pack(side="left",anchor="nw")

        for row in self.row_frames:
            row.pack(side="top", anchor="nw")
        self.m_equals_text_frame.pack(side="top")

        self.button_frame = ttk.Frame(self.win)

        self.clear_matrix_button = ttk.Button(self.button_frame, text="Clear Matrix")
        self.clear_matrix_button.pack(side="left")

        self.normalise_matrix_button = ttk.Button(self.button_frame,text="Normalise M (Set det M = 1)")
        self.normalise_matrix_button.pack(side="left", padx=25)

        self.apply_matrix_button = ttk.Button(self.button_frame, text="Apply Matrix")
        self.apply_matrix_button.pack(side="left", padx=(0,25))
        self.close_button = ttk.Button(self.button_frame, text="Close")
        self.close_button.pack(side="left")


        self.button_frame.pack(side="top", pady=(20,0))

        self.matrix_frame.pack(side="left", padx=(5,25), pady=5)

        self.error_variable = tk.StringVar()
        self.error_label = tk.Label(self.win, textvariable=self.error_variable, fg="red")
        self.error_label.pack(side="top", pady=5)

        #self.win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))

        self.matrix_variables = []
        i = 0
        for entry in self.matrix_entries:
            string_variable = tk.StringVar()
            self.matrix_variables.append(string_variable)
            entry["textvariable"] = string_variable

            if i in [0, 4, 8]:
                string_variable.set("1")
            else:
                string_variable.set("0")
            i += 1

        app.m_matrix_data = self.matrix_variables

        self.clear_matrix_button.bind("<ButtonPress>", self.clear_matrix)
        self.apply_matrix_button.bind("<ButtonPress>", self.apply_matrix)
        self.normalise_matrix_button.bind("<ButtonPress>", self.normalise_matrix)
        self.close_button.bind("<ButtonPress>", lambda e: self.win.destroy())

    def apply_matrix(self,event):

        try:
            M = self.create_matrix()
            self.error_variable.set("")
        except:
            self.error_variable.set("One or more entries are not well-defined.")
            return


        try:

            assert np.isclose(np.linalg.det(M),1)
            M_inverse = np.linalg.inv(M)
            vertex_dictionary = {}
            for triangle in app.main_surface.triangles:
                for vertex in triangle.vertices:
                    try:
                        vertex_dictionary[vertex]
                    except:
                        vertex_dictionary[vertex] = 1

            vertices = list(vertex_dictionary.keys())
            for vertex in vertices:
                vertex.r = np.matmul(vertex.r,M_inverse)
                vertex.c = np.matmul(M,vertex.c)
                vertex.r_clover = np.matmul(vertex.r_clover, M_inverse)
                vertex.c_clover = np.matmul(M, vertex.c_clover)
            self.error_variable.set("")
            app.plot_fresh(app.t)
        except:
            self.error_variable.set("The matrix does not have determinant 1. Please normalise the matrix first.")

    def create_matrix(self):
        M = []
        matrix_data = [app.string_fraction_to_float(string.get()) for string in self.matrix_variables]
        matrix_data = matrix_data[::-1]
        for i in range(3):
            row = []
            for j in range(3):
                row.append(matrix_data.pop())
            M.append(row)
        return np.array(M)

    def clear_matrix(self, event):
        for var in self.matrix_variables:
            var.set("")
        self.error_variable.set("")

    def normalise_matrix(self, event):
        try:
            self.create_matrix()
            self.error_variable.set("")
        except:
            self.error_variable.set("One or more entries are not well-defined.")
            return
        try:
            assert not np.isclose(np.linalg.det(self.create_matrix()), 0)
            determinant = np.linalg.det(self.create_matrix())
            cube_root_determinant = np.sign(determinant)*np.power(abs(determinant),(1/3))
            for var in self.matrix_variables:
                var.set(f'{app.string_fraction_to_float(var.get())/cube_root_determinant}')
            self.error_variable.set("")
        except:
            self.error_variable.set("This matrix is singular and has determinant zero. Please enter a non-singular matrix.")


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(anchor='nw')
        self.plot_data = []
        self.left_side_frame = ttk.Frame()
        self.abstract_surface = None
        self.create_surface_frame = ttk.Frame(self.left_side_frame)
        self.triangle_parameter_label = ttk.Label(self.create_surface_frame,justify='left', text='Set Initial Triangle Parameter (Must be positive)')
        self.triangle_parameter_label.pack(side='top', anchor='nw',padx=25, pady=10)
        self.entry_parameter = ttk.Entry(self.create_surface_frame,justify='left')
        self.entry_parameter.pack(side='top', anchor='nw', padx=25, pady=5)
        self.half_edge_param_label = ttk.Label(self.create_surface_frame,justify='left', text='Set Initial Half-Edge eij Parameters [e01,e10, e02, e20, e12, e21] (Must be positive)')
        self.half_edge_param_label.pack(side='top', anchor='nw', padx=25, pady=10)
        self.half_edge_param_entries = []
        for i in range(6):
            self.half_edge_param_entries.append(ttk.Entry(self.create_surface_frame,width=5))
        for half_edge_param_entry in self.half_edge_param_entries:
            half_edge_param_entry.pack(side='left',anchor='nw', padx=(25,8) if half_edge_param_entry == self.half_edge_param_entries[0] else 8, pady=0)

        self.create_surface_frame.pack(side='top', anchor='nw')

        self.surface_buttons_frame = ttk.Frame(self.left_side_frame)
        self.randomise_button = ttk.Button(self.surface_buttons_frame,text='Randomise Numbers')
        self.randomise_button.pack(side='left',anchor='nw', padx=25, pady=25)

        

        self.add_initial_triangle_button = ttk.Button(self.surface_buttons_frame,text='Add Initial Triangle')
        self.add_initial_triangle_button.pack(side='left',anchor='nw', padx=20, pady=25)
        self.surface_buttons_frame.pack(side='top', anchor='nw')

        self.error_text = tk.StringVar()
        self.error_text.set("")
        self.error_message_label = tk.Label(self.left_side_frame,justify='left',textvariable=self.error_text, fg='red')

        self.error_message_label.pack(side='left',anchor='nw', padx=25, pady=0)

        self.add_triangle_frame = ttk.Frame(self.left_side_frame)

        self.add_triangle_param_label = ttk.Label(self.add_triangle_frame, justify='right',
                                               text='Set Parameters In New Triangle [e03, e30, e23, e32, A023] (Must be positive)')
        self.add_triangle_param_label.pack(side='top', anchor='nw', padx=25, pady=10)
        self.add_triangle_param_entries = []
        for i in range(5):
            self.add_triangle_param_entries.append(ttk.Entry(self.add_triangle_frame, width=5))
        for add_triangle_param_entry in self.add_triangle_param_entries:
            add_triangle_param_entry.pack(side='left', anchor='nw',
                                       padx=(25, 8) if add_triangle_param_entry == self.add_triangle_param_entries[0] else 8,
                                       pady=0)

        self.add_triangle_frame.pack(side='top', anchor='nw')

        self.add_triangle_button_frame = ttk.Label(self.left_side_frame)
        self.add_triangle_randomise_button = ttk.Button(self.add_triangle_button_frame, text='Randomise Numbers')
        self.add_triangle_randomise_button.pack(side='left', anchor='nw', padx=25, pady=25)

        self.add_triangle_button = ttk.Button(self.add_triangle_button_frame, text='Add New Triangle To Selected Edge')
        self.add_triangle_button.pack(side='left', anchor='nw', padx=20, pady=25)

        self.add_triangle_button_frame.pack(side='top',anchor='nw')


        self.add_triangle_error_text = tk.StringVar()
        self.add_triangle_error_text.set("")
        self.add_triangle_error_message_label = tk.Label(self.left_side_frame, justify='left', textvariable=self.add_triangle_error_text,
                                            fg='red')

        self.add_triangle_error_message_label.pack(side='top', anchor='nw', padx=25, pady=0)

        self.plot_buttons_frame = ttk.Frame(self.left_side_frame)

        self.canonical_cell_decomp_button = ttk.Button(self.plot_buttons_frame, text='Canonical Cell Decomposition')
        self.canonical_cell_decomp_button.pack(side='left', anchor='nw', padx=5, pady=0)



        self.generate_surface = ttk.Button(self.plot_buttons_frame, text='Generate Hypersurface')
        self.generate_surface.pack(side='left', anchor='nw', padx=5, pady=0)
        self.generate_surface_s3 = ttk.Button(self.plot_buttons_frame, text='Generate Hypersurface (Projected S³)')
        self.generate_surface_s3.pack(side='left', anchor='nw', padx=5, pady=0)
        self.plot_buttons_frame.pack(side='top',anchor='nw')
        self.generate_surface_error_text = tk.StringVar()
        self.generate_surface_error_text.set("")
        self.generate_surface_error_message_label = tk.Label(self.left_side_frame, justify='left',
                                                         textvariable=self.generate_surface_error_text,
                                                         fg='red')

        self.generate_surface_error_message_label.pack(side='top', anchor='nw', padx=25, pady=5)


        self.left_side_frame.pack(side='left',anchor='nw')
        # Create the application variable.
        self.triangle_parameter = tk.StringVar()
        self.half_edge_params = []
        for half_edge_param_entry in self.half_edge_param_entries:
            self.half_edge_params.append(tk.StringVar())
            self.half_edge_params[-1].set(1)
            half_edge_param_entry["textvariable"] = self.half_edge_params[-1]

        self.add_triangle_params = []
        for add_triangle_param_entry in self.add_triangle_param_entries:
            self.add_triangle_params.append(tk.StringVar())
            self.add_triangle_params[-1].set(1)
            add_triangle_param_entry["textvariable"] = self.add_triangle_params[-1]


        # Set it to some value.
        self.triangle_parameter.set("1")
        # Tell the entry widget to watch this variable.
        self.entry_parameter["textvariable"] = self.triangle_parameter





        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_type = FigureCanvasTkAgg(self.figure, root)
        self.chart_type.get_tk_widget().pack()
        self.ax.set_title('Cloverleaf Position')
        self.ax.set_axis_off()

        # Define a callback for when the user hits return.
        # It prints the current value of the variable.
        self.add_initial_triangle_button.bind('<ButtonPress>',
                             self.add_initial_triangle)
        self.randomise_button.bind('<ButtonPress>',
                            self.randomise_numbers_initial)
        self.add_triangle_randomise_button.bind('<ButtonPress>',
                                   self.randomise_numbers_add_triangle)
        self.add_triangle_button.bind('<ButtonPress>',
                                      self.add_triangle)
        self.generate_surface.bind('<ButtonPress>',
                                      self.generate_surface_visual)
        self.generate_surface_s3.bind('<ButtonPress>',
                                   self.generate_surface_s3_function)

        self.canonical_cell_decomp_button.bind('<ButtonPress>',
                                        self.canonical_cell_decomp)

    





    def generate_new_triangle(self,current_edge, current_abstract_edge, distance_from_initial_triangle, e03, e30, e23, e32, A023, max_distance):

        


        if distance_from_initial_triangle > max_distance:
            return

        if len(self.abstract_surface.triangles) == len(self.reduced_main_surface.triangles):
            return

        v0, v1, flipped = app.correct_edge_orientation(current_edge)

        r3, c3 = compute_all_until_r3c3(v0.r, v1.r, v0.c,
                                              v1.c, e03, e23,  e30, e32, A023)

        r3_clover, c3_clover = compute_all_until_r3c3(v0.r_clover, v1.r_clover, v0.c_clover,
                                              v1.c_clover, e03, e23,  e30, e32, A023)

        
        
        new_triangle = self.reduced_main_surface.add_triangle(current_edge, v0, v1, Vertex(c3, r3, c3_clover, r3_clover))
        
        abstract_triangle = current_abstract_edge.edge_glued[2].triangle

        new_triangle.index = abstract_triangle.index
        

        edge_index = 0
        for index in range(3):
            if abstract_triangle.edges[index] == current_abstract_edge.edge_glued[2]:
                edge_index = index

        new_triangle.edges[0].abstract_index = current_abstract_edge.edge_glued[2].index
        new_triangle.edges[1].abstract_index = abstract_triangle.edges[(edge_index+1)%3].index
        new_triangle.edges[2].abstract_index = abstract_triangle.edges[(edge_index-1)%3].index

        next_edge_indices = [1,-1]


        next_surface_index = 0
        for next_surface_edge in new_triangle.edges[1:]:
            next_abstract_edge = abstract_triangle.edges[(edge_index+next_edge_indices[next_surface_index])%3]
            next_surface_index +=1
            edge_glued = next_abstract_edge.edge_glued[2]
            if edge_glued.triangle.index in [triangle.index for triangle in self.reduced_main_surface.triangles]:
                continue
            edge_glued_index = 0
            for index in range(3):
                if edge_glued.triangle.edges[index] == edge_glued:
                    edge_glued_index = index

            edge_forward = edge_glued.triangle.edges[(edge_glued_index + 1) % 3]
            edge_backward = edge_glued.triangle.edges[(edge_glued_index - 1) % 3]

            if edge_backward.index == '02':
                e03 = edge_backward.ea
                e30 = edge_backward.eb
            else:
                e03 = edge_backward.eb
                e30 = edge_backward.ea
            if edge_forward.index == '02':
                e32 = edge_forward.ea
                e23 = edge_forward.eb
            else:
                e32 = edge_forward.eb
                e23 = edge_forward.ea
            A023 = edge_glued.triangle.triangle_parameter

            next_surface_edge.abstract_index = next_abstract_edge.index
            
            self.generate_new_triangle(next_surface_edge, next_abstract_edge,
                                  distance_from_initial_triangle+1, e03, e30, e23, e32, A023, max_distance)

        return

    def generate_all_ones_triangle(self, current_edge,current_triangle):
        for edge in current_triangle.edges:
            if edge == current_edge:
                continue
            if edge.connected:
                v0, v1, flipped = app.correct_edge_orientation(edge)
                r3,c3 = compute_all_until_r3c3(v0.r,v1.r,v0.c,v1.c,1,1,1,1,1)
                next_triangle = edge.edge_connected.triangle
                next_edge = next_triangle.edges[(edge.edge_connected.index+1)%3]
                next_triangle.t = 1
                next_edge.v1.c = c3
                next_edge.v1.r = r3
                next_edge.v1.c_clover = c3
                next_edge.v1.r_clover = r3
                self.generate_all_ones_triangle(edge.edge_connected,next_triangle)



    def compute_centre_cell_decomp(self):
        try:
            initial_triangle_index = 0
            max_distance = len(self.abstract_surface.triangles)

            initial_abstract_triangle = self.abstract_surface.triangles[0]
            for triangle in self.abstract_surface.triangles:
                if triangle.index == initial_triangle_index:
                    initial_abstract_triangle = triangle

            t = initial_abstract_triangle.triangle_parameter
            e01 = initial_abstract_triangle.edges[0].ea
            e02 = initial_abstract_triangle.edges[2].ea
            e10 = initial_abstract_triangle.edges[0].eb
            e12 = initial_abstract_triangle.edges[1].ea
            e20 = initial_abstract_triangle.edges[2].eb
            e21 = initial_abstract_triangle.edges[1].eb

            cube_root_a_coord_t = np.power(t, (1 / 3))
            c0 = [cube_root_a_coord_t, 0, 0]
            c1 = [0, cube_root_a_coord_t, 0]
            c2 = [0, 0, cube_root_a_coord_t]
            r0 = [0, e01 / cube_root_a_coord_t, e02 / cube_root_a_coord_t]
            r1 = [e10 / cube_root_a_coord_t, 0, e12 / cube_root_a_coord_t]
            r2 = [e20 / cube_root_a_coord_t, e21 / cube_root_a_coord_t, 0]

            c0_clover = [1, 0, 0]
            c1_clover = [0, 1, 0]
            c2_clover = [0, 0, 1]

            x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
            cube_root_x_coord_t = np.power(x_coord_t, 1 / 3)

            r0_clover = [0, cube_root_x_coord_t, 1]
            r1_clover = [1, 0, cube_root_x_coord_t]
            r2_clover = [cube_root_x_coord_t, 1, 0]

            self.reduced_main_surface = Surface(c0, c1, c2, r0, r1, r2, c0_clover, c1_clover, c2_clover, r0_clover, r1_clover,
                                        r2_clover)

            self.reduced_main_surface.triangles[0].index = initial_abstract_triangle.index
            self.reduced_main_surface.triangles[0].t = cube_root_a_coord_t

            for edge_index in range(3):
                edge = initial_abstract_triangle.edges[edge_index]
                edge_glued = initial_abstract_triangle.edges[edge_index].edge_glued[2]
                edge_glued_index = 0
                for index in range(3):
                    if edge_glued.triangle.edges[index] == edge_glued:
                        edge_glued_index = index

                edge_forward = edge_glued.triangle.edges[(edge_glued_index + 1) % 3]
                edge_backward = edge_glued.triangle.edges[(edge_glued_index - 1) % 3]

                if edge_backward.index == '02':
                    e03 = edge_backward.ea
                    e30 = edge_backward.eb
                else:
                    e03 = edge_backward.eb
                    e30 = edge_backward.ea
                if edge_forward.index == '02':
                    e32 = edge_forward.ea
                    e23 = edge_forward.eb
                else:
                    e32 = edge_forward.eb
                    e23 = edge_forward.ea

                A023 = edge_glued.triangle.triangle_parameter

                
                self.reduced_main_surface.triangles[0].edges[edge_index].abstract_index = edge.index
                self.generate_new_triangle(self.reduced_main_surface.triangles[0].edges[edge_index], edge, 0, e03, e30, e23,
                                           e32, A023, max_distance)
            for triangle in self.reduced_main_surface.triangles.copy():
                for edge_index in range(3):
                    edge = triangle.edges[edge_index]
                    if not edge.connected:
                        abstract_triangle = self.abstract_surface.triangles[triangle.index]
                        for abstract_edge in abstract_triangle.edges:
                            if abstract_edge.index == edge.abstract_index:
                                edge_glued = abstract_edge.edge_glued[2]
                                edge_glued_index = 0
                                for index in range(3):
                                    if edge_glued.triangle.edges[index] == edge_glued:
                                        edge_glued_index = index

                                edge_forward = edge_glued.triangle.edges[(edge_glued_index + 1) % 3]
                                edge_backward = edge_glued.triangle.edges[(edge_glued_index - 1) % 3]

                                if edge_backward.index == '02':
                                    e03 = edge_backward.ea
                                    e30 = edge_backward.eb
                                else:
                                    e03 = edge_backward.eb
                                    e30 = edge_backward.ea
                                if edge_forward.index == '02':
                                    e32 = edge_forward.ea
                                    e23 = edge_forward.eb
                                else:
                                    e32 = edge_forward.eb
                                    e23 = edge_forward.ea
                                
                                current_edge = edge

                                A023 = edge_glued.triangle.triangle_parameter
                                
                                v0, v1, flipped = app.correct_edge_orientation(current_edge)

                                r3, c3 = compute_all_until_r3c3(v0.r, v1.r, v0.c,
                                                                    v1.c, e03, e23,  e30, e32, A023)


                                r3_clover, c3_clover = compute_all_until_r3c3(v0.r_clover, v1.r_clover, v0.c_clover,
                                                                    v1.c_clover, e03, e23,  e30, e32, A023)

                                new_triangle = self.reduced_main_surface.add_triangle(current_edge, v0, v1, Vertex(c3, r3, c3_clover, r3_clover))
                                new_triangle.index = edge_glued.triangle.index


                                current_edge.edge_connected.abstract_index = edge_glued.index
                                new_triangle.edges[(current_edge.edge_connected.index+1)%3].abstract_index = edge_forward.index
                                new_triangle.edges[(current_edge.edge_connected.index-1)%3].abstract_index = edge_backward.index

            
            edge_flip_sequence = []

            found_no_edges = False
            while not found_no_edges:
                found_edge = False
                for triangle in self.reduced_main_surface.triangles:
                    if not found_edge:
                        for edge in triangle.edges:
                            if edge.edge_connected:
                                c0 = edge.v0.c
                                edge_connected = edge.edge_connected
                                c1 = edge_connected.triangle.edges[(edge_connected.index + 1) % 3].v1.c
                                c2 = edge.v1.c
                                c3 = edge.triangle.edges[(edge.index + 1) % 3].v1.c
                                outitude_sign = compute_outitude_sign(c0, c1, c2, c3)
                                if outitude_sign < 0:
                                    r0 = edge.v0.r
                                    r1 = edge_connected.triangle.edges[(edge_connected.index + 1) % 3].v1.r
                                    r2 = edge.v1.r
                                    r3 = edge.triangle.edges[(edge.index + 1) % 3].v1.r
                                    e01 = np.dot(r0,c1)
                                    e10 = np.dot(r1, c0)
                                    e12 = np.dot(r1, c2)
                                    e21 = np.dot(r2,c1)
                                    e02 = np.dot(r0,c2)
                                    e20 = np.dot(r2,c0)
                                    print('edge outitude: ',outitude_edge_params(7,8,e10,e01, e21, e12, e02, e20))
                                    print(e01,e10,e12,e21,e20,e02)


                                    e_prime = self.reduced_main_surface.flip_edge(edge)
                                    
                                    e_prime_forward = e_prime.triangle.edges[(e_prime.index+1)%3]
                                    e_prime_backward = e_prime.triangle.edges[(e_prime.index-1)%3]
                                    
                                    if e_prime_backward.abstract_index != '02':
                                        forward_sorted = np.sort([(int(e_prime_backward.abstract_index[0])-1)%3,int(e_prime_backward.abstract_index[0])])
                                    else:
                                        forward_sorted = np.sort([(int(e_prime_backward.abstract_index[1])-1)%3,int(e_prime_backward.abstract_index[1])])
                                    e_prime_forward.abstract_index = f'{forward_sorted[0]}{forward_sorted[1]}'
                                    prime_sorted = []
                                    if e_prime_backward.abstract_index != '02':
                                        prime_sorted.append(e_prime_backward.abstract_index[1])
                                    else:
                                        prime_sorted.append(e_prime_backward.abstract_index[0])
                                    if e_prime_forward.abstract_index != '02':
                                        prime_sorted.append(e_prime_forward.abstract_index[0])
                                    else:
                                        prime_sorted.append(e_prime_forward.abstract_index[1])
                                    prime_sorted = np.sort(prime_sorted)
                                    e_prime.abstract_index = f'{prime_sorted[0]}{prime_sorted[1]}'
                                    e_prime_connected = e_prime.edge_connected
                                    e_prime_connected_forward = e_prime_connected.triangle.edges[(e_prime_connected.index+1)%3]
                                    e_prime_connected_backward = e_prime_connected.triangle.edges[(e_prime_connected.index-1)%3]
                                    
                                    if e_prime_connected_backward.abstract_index != '02':
                                        forward_sorted = np.sort([(int(e_prime_connected_backward.abstract_index[0])-1)%3,int(e_prime_connected_backward.abstract_index[0])])
                                    else:
                                        forward_sorted = np.sort([(int(e_prime_connected_backward.abstract_index[1])-1)%3,int(e_prime_connected_backward.abstract_index[1])])
                                    e_prime_connected_forward.abstract_index = f'{forward_sorted[0]}{forward_sorted[1]}'
                                    prime_sorted = []
                                    if e_prime_connected_backward.abstract_index != '02':
                                        prime_sorted.append(e_prime_connected_backward.abstract_index[1])
                                    else:
                                        prime_sorted.append(e_prime_connected_backward.abstract_index[0])
                                    if e_prime_connected_forward.abstract_index != '02':
                                        prime_sorted.append(e_prime_connected_forward.abstract_index[0])
                                    else:
                                        prime_sorted.append(e_prime_connected_forward.abstract_index[1])
                                    prime_sorted = np.sort(prime_sorted)
                                    e_prime_connected.abstract_index = f'{prime_sorted[0]}{prime_sorted[1]}'
                                    edge_flip_sequence.append((edge,e_prime))
                                    
                                    found_edge = True
                                    break
                if not found_edge:
                    found_no_edges = True
                
            print([(x.abstract_index,x.triangle.index, y.abstract_index, y.triangle.index) for x,y in edge_flip_sequence])
            print(len(edge_flip_sequence))
            
            self.reduced_main_surface.triangles[0].vertices[0].c = [1,0,0]
            self.reduced_main_surface.triangles[0].vertices[1].c = [0, 1, 0]
            self.reduced_main_surface.triangles[0].vertices[2].c = [0, 0, 1]
            self.reduced_main_surface.triangles[0].vertices[0].r = [0, 1, 1]
            self.reduced_main_surface.triangles[0].vertices[1].r = [1, 0, 1]
            self.reduced_main_surface.triangles[0].vertices[2].r = [1, 1, 0]
            self.reduced_main_surface.triangles[0].vertices[0].c_clover = [1, 0, 0]
            self.reduced_main_surface.triangles[0].vertices[1].c_clover = [0, 1, 0]
            self.reduced_main_surface.triangles[0].vertices[2].c_clover = [0, 0, 1]
            self.reduced_main_surface.triangles[0].vertices[0].r_clover = [0, 1, 1]
            self.reduced_main_surface.triangles[0].vertices[1].r_clover = [1, 0, 1]
            self.reduced_main_surface.triangles[0].vertices[2].r_clover = [1, 1, 0]
            self.reduced_main_surface.triangles[0].t = 1


            self.generate_all_ones_triangle(None,self.reduced_main_surface.triangles[0])


            while edge_flip_sequence:
                _, next_flip_edge = edge_flip_sequence.pop()
                e_prime = self.reduced_main_surface.flip_edge(next_flip_edge.edge_connected)                                                     
                e_prime_forward = e_prime.triangle.edges[(e_prime.index+1)%3]
                e_prime_backward = e_prime.triangle.edges[(e_prime.index-1)%3]   
                if e_prime_forward.abstract_index != '02':
                    backward_sorted = np.sort([(int(e_prime_forward.abstract_index[0])-1)%3,int(e_prime_forward.abstract_index[1])])
                else:
                    backward_sorted = np.sort([(int(e_prime_forward.abstract_index[1])-1)%3,int(e_prime_forward.abstract_index[0])])
                e_prime_backward.abstract_index = f'{backward_sorted[0]}{backward_sorted[1]}'
                prime_sorted = []
                if e_prime_forward.abstract_index != '02':
                    prime_sorted.append(e_prime_forward.abstract_index[0])
                else:
                    prime_sorted.append(e_prime_forward.abstract_index[1])
                if e_prime_backward.abstract_index != '02':
                    prime_sorted.append(e_prime_backward.abstract_index[1])
                else:
                    prime_sorted.append(e_prime_backward.abstract_index[0])
                prime_sorted = np.sort(prime_sorted)
                e_prime.abstract_index = f'{prime_sorted[0]}{prime_sorted[1]}'
                e_prime_connected = e_prime.edge_connected
                e_prime_connected_forward = e_prime_connected.triangle.edges[(e_prime_connected.index+1)%3]
                e_prime_connected_backward = e_prime_connected.triangle.edges[(e_prime_connected.index-1)%3]
                if e_prime_connected_forward.abstract_index != '02':
                    backward_sorted = np.sort([(int(e_prime_connected_forward.abstract_index[0])-1)%3,int(e_prime_connected_forward.abstract_index[1])])
                else:
                    backward_sorted = np.sort([(int(e_prime_connected_forward.abstract_index[1])-1)%3,int(e_prime_connected_forward.abstract_index[0])])
                e_prime_connected_backward.abstract_index = f'{backward_sorted[0]}{backward_sorted[1]}'
                prime_sorted = []
                if e_prime_connected_forward.abstract_index != '02':
                    prime_sorted.append(e_prime_connected_forward.abstract_index[0])
                else:
                    prime_sorted.append(e_prime_connected_forward.abstract_index[1])
                if e_prime_connected_backward.abstract_index != '02':
                    prime_sorted.append(e_prime_connected_backward.abstract_index[1])
                else:
                    prime_sorted.append(e_prime_connected_backward.abstract_index[0])
                prime_sorted = np.sort(prime_sorted)
                e_prime_connected.abstract_index = f'{prime_sorted[0]}{prime_sorted[1]}'
                [e_prime.triangle.index,e_prime_connected.triangle.index] = [e_prime_connected.triangle.index, e_prime.triangle.index]
                prime_triangle_list_index = 0
                prime_connected_triangle_list_index = 0
                for index in range(len(self.reduced_main_surface.triangles)):
                    if self.reduced_main_surface.triangles[index] == e_prime.triangle:
                        prime_triangle_list_index = index
                    if self.reduced_main_surface.triangles[index] == e_prime_connected.triangle:
                        prime_connected_triangle_list_index = index
                
                self.reduced_main_surface.triangles[prime_triangle_list_index] = e_prime_connected.triangle
                self.reduced_main_surface.triangles[prime_connected_triangle_list_index] = e_prime.triangle
                
                
                edge = next_flip_edge.edge_connected
                edge_forward = edge.triangle.edges[(edge.index+1)%3]
                edge_backward = edge.triangle.edges[(edge.index-1)%3]
                edge_connected = edge.edge_connected
                e_minus = np.dot(edge.v1.r, edge.v0.c)
                e_plus = np.dot(edge.v0.r, edge.v1.c)
                c_plus = np.dot(edge_forward.v0.r,edge_forward.v1.c)
                b_plus = np.dot(e_prime_forward.v1.r, e_prime_forward.v0.c)
                d_minus = np.dot(edge_backward.v1.r,edge_backward.v0.c)
                a_minus = np.dot(e_prime_connected_backward.v0.r,e_prime_connected_backward.v1.c)
                A = edge_connected.triangle.t
                B = edge.triangle.t
                C = (B*c_plus + A*b_plus)/e_minus
                D = (A*d_minus+B*a_minus)/e_plus
                e_prime.triangle.t = C
                e_prime_connected.triangle.t = D
                            
            self.canonical_abstract_surface = AbstractSurface()
            for triangle in self.reduced_main_surface.triangles[:len(self.abstract_surface.triangles)]:
                self.canonical_abstract_surface.add_triangle()
                self.canonical_abstract_surface.triangles[-1].index = triangle.index     

            self.canonical_abstract_surface.triangles = np.array(self.canonical_abstract_surface.triangles)[np.argsort([triangle.index for triangle in self.canonical_abstract_surface.triangles])]      
            
            

            for triangle in self.reduced_main_surface.triangles[:len(self.abstract_surface.triangles)]:
                for edge in triangle.edges:
                    abstract_edge = self.canonical_abstract_surface.triangles[triangle.index].edges[0]
                    for abstract_edge_index in range(3):
                        if self.canonical_abstract_surface.triangles[triangle.index].edges[abstract_edge_index].index == edge.abstract_index:
                            abstract_edge = self.canonical_abstract_surface.triangles[triangle.index].edges[abstract_edge_index]
                    abstract_triangle_original = self.abstract_surface.triangles[triangle.index]
                    for temp_edge in abstract_triangle_original.edges:
                        if temp_edge.index == edge.abstract_index:    
                            abstract_edge_original = temp_edge
                    for temp_edge in self.canonical_abstract_surface.triangles[abstract_edge_original.edge_glued[2].triangle.index].edges:
                        if temp_edge.index == abstract_edge_original.edge_glued[2].index:
                            abstract_edge_glued = temp_edge
                    
                    flipped = (abstract_edge_original.edge_glued[1] != abstract_edge_original.edge_glued[2].v0)

                    if not flipped:
                        self.canonical_abstract_surface.glue_edges(abstract_edge, abstract_edge_glued, abstract_edge.v0, abstract_edge_glued.v0)
                    else:
                        self.canonical_abstract_surface.glue_edges(abstract_edge, abstract_edge_glued, abstract_edge.v0, abstract_edge_glued.v1)

                    abstract_edge.ea = np.dot(edge.v0.r,edge.v1.c)
                    abstract_edge.eb = np.dot(edge.v1.r, edge.v0.c)
                   
                    
                    abstract_edge.triangle.triangle_parameter = edge.triangle.t

            dir_name = filedialog.asksaveasfilename(filetypes=[("Excel files", ".csv")])
            if not dir_name:
                return

            if '.csv' in dir_name:
                dir_name = dir_name[:-4]


            gluing_table_data = []
            
            for triangle in self.canonical_abstract_surface.triangles:
                row = [triangle.index]
                for edge in triangle.edges:
                    flipped = (edge.edge_glued[1]!=edge.edge_glued[2].v0)
                    if not flipped:
                        row.append(f"{edge.edge_glued[2].triangle.index} ({edge.edge_glued[2].index})")
                    else:
                        row.append(f"{edge.edge_glued[2].triangle.index} ({edge.edge_glued[2].index[::-1]})")
                gluing_table_data.append(row)
            

            gluing_table_data = np.array(gluing_table_data)
            gluing_table_data_last_col = gluing_table_data[:,-1].copy()
            gluing_table_data[:,-1] = gluing_table_data[:,-2]
            gluing_table_data[:,-2] = gluing_table_data_last_col

            parameter_table_data = []

            for triangle in self.canonical_abstract_surface.triangles:
                row = [triangle.triangle_parameter]
                for edge in triangle.edges:
                    if edge.index != '02':
                        row.append(edge.ea)
                    else:
                        row.append(edge.eb)
                parameter_table_data.append(row)
            
            parameter_table_data = np.array(parameter_table_data)
            parameter_table_data_last_col = parameter_table_data[:,-1].copy()
            parameter_table_data[:,-1] = parameter_table_data[:,-2]
            parameter_table_data[:,-2] = parameter_table_data_last_col

            column_names = np.array(['Triangle', 'Edge 01', 'Edge 02', 'Edge 12', 'Triangle Parameter', 'Edge 01', 'Edge 02', 'Edge 12'])
            data = np.hstack([gluing_table_data,parameter_table_data])
            table = pd.DataFrame(data)
            table.columns = column_names
            table.to_csv(f"{dir_name}.csv", index=False)




            self.generate_surface_error_text.set(
                "")
        except:
            self.generate_surface_error_text.set(
                "Please import a gluing table before computing centre coordinates of canonical cell decomposition.")


        pass


    def canonical_cell_instructions(self, event):
        pass


    def canonical_cell_decomp(self, event):
        
        try:
            found_no_edges = False
            while not found_no_edges:
                found_edge = False
                for triangle in self.main_surface.triangles:
                    if not found_edge:
                        for edge in triangle.edges:
                            if edge.edge_connected:
                                c0 = edge.v0.c
                                edge_connected = edge.edge_connected
                                c1 = edge_connected.triangle.edges[(edge_connected.index+1) % 3].v1.c
                                c2 = edge.v1.c
                                c3 = edge.triangle.edges[(edge.index+1) % 3].v1.c
                                outitude_sign = compute_outitude_sign(c0,c1,c2,c3)
                                if outitude_sign < 0:
                                    self.main_surface.flip_edge(edge)
                                    found_edge = True
                                    break
                if not found_edge:
                    found_no_edges = True
            
            self.plot_fresh(self.t)
            self.generate_surface_error_text.set("")
        except:
            self.generate_surface_error_text.set("Please add an initial triangle before computing canonical cell decomposition.")


    def plot_fresh(self, t):
        self.t = t
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.remove()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Cloverleaf Position')
        self.ax.set_axis_off()
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.edge_selected = self.main_surface.triangles[-1].edges[-2]
        if self.edge_selected.connected:
            for edge in self.main_surface.triangles[-1].edges:
                if not edge.connected:
                    self.edge_selected = edge
        for triangle in self.main_surface.triangles:
            [x1, y1, z1] = triangle.vertices[0].c_clover
            [x2, y2, z2] = triangle.vertices[1].c_clover
            [x3, y3, z3] = triangle.vertices[2].c_clover
            [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
            [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
            [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)
            x = [x1, x2, x3, x1]
            y = [y1, y2, y3, y1]

            self.plot_data.append(self.ax.plot(x, y, c='blue'))
        v0 = self.edge_selected.v0.c_clover
        v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
        v1 = self.edge_selected.v1.c_clover
        v1 = clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t)
        self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                                           [v0[1], v1[1]], c='red'))
        self.chart_type.draw()
        self.generate_surface_error_text.set("")


    def generate_surface_s3_function(self, event):

        try:
            self.generate_surface_error_text.set("")
            surface_vis = SurfaceVisual(self.main_surface)
            surface_vis.show_vis_projected_3d()
        except:
            self.generate_surface_error_text.set("Please add an initial triangle before generating hypersurface (projected S³).")

    def string_fraction_to_float(self, string):
        if '/' in string:
            string = string.rsplit('/')
            return float(string[0])/float(string[1])
        return float(string)

    def onclick(self,event):
        coord = np.array([event.xdata,event.ydata])
        if not coord[0] or not coord[1]:
            return
        all_edges = []
        for triangle in self.main_surface.triangles:
            for edge in triangle.edges:
                if not edge.connected:
                    all_edges.append(edge)
        distances = []
        for edge in all_edges:
            v0 = edge.v0.c_clover
            v0 = np.array(clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t))
            v1 = edge.v1.c_clover
            v1 = np.array(clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t))
            v = v1-v0
            x = coord - v0
            distances.append(np.linalg.norm(x - abs(np.dot(x,v))/(np.linalg.norm(v)**2) * v))
        if self.edge_selected:
            self.plot_data[-1][0].remove()
        self.edge_selected = all_edges[np.argmin(distances)]

        v0 = self.edge_selected.v0.c_clover
        v0 = clover_position([[v0[0]],[v0[1]],[v0[2]]], self.t)
        v1 = self.edge_selected.v1.c_clover
        v1 = clover_position([[v1[0]],[v1[1]],[v1[2]]], self.t)
        self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                     [v0[1], v1[1]],c='red'))

        self.chart_type.draw()

    def correct_edge_orientation(self, edge):
        [v0,v1,v2] = edge.triangle.vertices
        vertices = np.array([v0, v1, v2])
        [v0, v1] = [edge.v0, edge.v1]
        flipped = False
        if vertices[(np.argwhere(edge.v0 == vertices)[0,0]+1)%3] == edge.v1:
            [v0, v1] = [v1,v0]

        return (v0, v1,flipped)

    def add_triangle(self, event):
        try:
            assert self.edge_selected
            e03 = self.string_fraction_to_float(self.add_triangle_params[0].get())
            e30 = self.string_fraction_to_float(self.add_triangle_params[1].get())
            e23 = self.string_fraction_to_float(self.add_triangle_params[2].get())
            e32 = self.string_fraction_to_float(self.add_triangle_params[3].get())
            A023 = self.string_fraction_to_float(self.add_triangle_params[4].get())
            assert e03 > 0 and e30 > 0 and e23 > 0 and e32 > 0 and A023 > 0

            v0, v1, flipped = self.correct_edge_orientation(self.edge_selected)


            r3, c3 = compute_all_until_r3c3(v0.r, v1.r, v0.c,
                                              v1.c, e03, e23, e30, e32, A023)

            r3_clover, c3_clover = compute_all_until_r3c3(v0.r_clover, v1.r_clover, v0.c_clover,
                                              v1.c_clover, e03, e23, e30, e32, A023)

            self.main_surface.add_triangle(self.edge_selected,v0,v1,Vertex(c3,r3, c3_clover, r3_clover))

            self.add_triangle_error_text.set("")
            if self.edge_selected:
                self.plot_data[-1][0].remove()
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title('Cloverleaf Position')
            self.ax.set_axis_off()
            self.figure.canvas.mpl_connect('button_press_event', self.onclick)
            self.edge_selected = self.main_surface.triangles[-1].edges[-2]

            for triangle in self.main_surface.triangles:
                [x1, y1, z1] = triangle.vertices[0].c_clover
                [x2, y2, z2] = triangle.vertices[1].c_clover
                [x3, y3, z3] = triangle.vertices[2].c_clover
                [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
                [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
                [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)
                x = [x1, x2, x3, x1]
                y = [y1, y2, y3, y1]
                self.plot_data.append(self.ax.plot(x, y,c='blue'))

            v0 = self.edge_selected.v0.c_clover
            v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
            v1 = self.edge_selected.v1.c_clover
            v1 = clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t)
            self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                                               [v0[1], v1[1]], c='red'))
            self.chart_type.draw()
        except:
            try:
                self.main_surface
                self.add_triangle_error_text.set("One or more variables are not well-defined.")
            except:
                self.add_triangle_error_text.set("Please add an initial triangle first.")


    def randomise_numbers_initial(self,event):
        for half_edge_param in self.half_edge_params:
            half_edge_param.set(round(abs(np.random.random()*10),1))
        param = round(abs(np.random.random()*10),1)
        while param == 0:
            param = round(abs(np.random.random()*10),1)
        self.triangle_parameter.set(param)

    def randomise_numbers_add_triangle(self,event):
        for add_triangle_param in self.add_triangle_params:
            add_triangle_param.set(round(np.random.random()*10,1))

    def add_initial_triangle(self, event):


        self.ax.clear()
        self.ax.set_axis_off()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Cloverleaf Position')
        self.ax.set_axis_off()
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        try:
            t = self.string_fraction_to_float(self.triangle_parameter.get())
            self.t = t
            e01 = self.string_fraction_to_float(self.half_edge_params[0].get())
            e10 = self.string_fraction_to_float(self.half_edge_params[1].get())
            e02 = self.string_fraction_to_float(self.half_edge_params[2].get())
            e20 = self.string_fraction_to_float(self.half_edge_params[3].get())
            e12 = self.string_fraction_to_float(self.half_edge_params[4].get())
            e21 = self.string_fraction_to_float(self.half_edge_params[5].get())
            assert t > 0 and e01 > 0 and e10 > 0 and e02 > 0 and e20 > 0 and e12 > 0 and e21 > 0

            c0 = [1,0,0]
            c1 = [0,t,0]
            c2 = [0,0,1]
            r0 = [0, e01 / t, e02]
            r1= [e10, 0, e12]
            r2= [e20, e21 / t, 0]
            c0_clover = [1, 0, 0]
            c1_clover = [0, 1, 0]
            c2_clover = [0, 0, 1]

            x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
            cube_root_x_coord_t = np.power(x_coord_t, (1 / 3))

            r0_clover = [0, cube_root_x_coord_t, 1]
            r1_clover = [1, 0, cube_root_x_coord_t]
            r2_clover = [cube_root_x_coord_t, 1, 0]
            self.error_text.set("")
            self.main_surface = Surface(c0, c1, c2, r0, r1, r2, c0_clover, c1_clover, c2_clover, r0_clover, r1_clover, r2_clover)
            self.edge_selected = self.main_surface.triangles[-1].edges[-1]

            for triangle in self.main_surface.triangles:
                [x1, y1, z1] = triangle.vertices[0].c_clover
                [x2, y2, z2] = triangle.vertices[1].c_clover
                [x3, y3, z3] = triangle.vertices[2].c_clover
                [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
                [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
                [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)

                x = [x1, x2, x3, x1]
                y = [y1, y2, y3, y1]

                self.plot_data.append(self.ax.plot(x, y,c='blue'))

            v0 = self.edge_selected.v0.c_clover
            v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
            v1 = self.edge_selected.v1.c_clover
            v1 = clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t)
            self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                                               [v0[1], v1[1]], c='red'))
            self.chart_type.draw()

        except:
            self.error_text.set("One or more variables are not well-defined.")






    def generate_surface_visual(self, event):
        try:
            self.generate_surface_error_text.set("")
            surface_vis = SurfaceVisual(self.main_surface)
            surface_vis.show_vis_3d()

        except:
            self.generate_surface_error_text.set("Please add an initial triangle before generating hypersurface.")

class CombinatorialImport:
    def __init__(self, tk, filename, create_window=True):
        if create_window:
            self.tk = tk
            self.parameter_entries = {}
            self.triangle_parameter_entry = None
            self.input_parameters = []
            self.convert_gluing_table_to_surface(filename)
            self.win = self.tk.Toplevel()
            self.win.resizable(width=False, height=False)
            self.win.wm_title("Uploaded Surface")
            self.l = tk.Label(self.win,
                         text="The uploaded gluing table is visualised as a combinatorial map below. You can change 𝒜-coordinate edge and triangle parameters by selecting a triangle. Once you're done, press continue.")
            self.l.pack(padx=20, pady=10)
            #self.win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
            self.figure = plt.Figure(figsize=(7, 5), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.chart_type = FigureCanvasTkAgg(self.figure, self.win)
            self.chart_type.get_tk_widget().pack()
            self.ax.set_title('Combinatorial Map')
            self.generate_combinatorial_map()
            self.ax.set_axis_off()
            self.chart_type.draw()
            self.cancel = ttk.Button(self.win, text="Cancel", command=self.win.destroy)
            self.cancel.pack(side='right', padx=25, pady=5)
            self.continue_button = ttk.Button(self.win, text="Continue",
                                         command=lambda: (self.generate_developing_map()))
            self.continue_button.pack(side='right', padx=10, pady=5)
            self.randomise_button = ttk.Button(self.win, text="Randomise Parameters", command=self.randomise_parameters)
            self.randomise_button.pack(side="left", padx=10, pady=5)

            self.depth_text=  tk.Label(self.win, text="Max Depth: ")
            self.depth_text.pack(side="left",padx=10,pady=5)
            self.depth_string = tk.StringVar()
            self.depth_string.set("5")
            

            self.depth_input = ttk.Entry(self.win, textvariable=self.depth_string, width=5)
            self.depth_input.pack(side="left", anchor="nw",padx=5,pady=25)
            


            self.error_text = tk.StringVar()
            self.error_text.set("")
            self.error_message_label = tk.Label(self.win, textvariable=self.error_text,
                                                fg='red')
            self.error_message_label.pack(side='left', padx=10, pady=5)
        else:
            f = open(f"{filename}", "rb")
            self.abstract_surface = pickle.load(f, encoding="bytes")
            self.abstract_surface.triangles
            f.close()
            self.generate_real_surface_map()
            app.main_surface = self.main_surface
            app.abstract_surface = self.abstract_surface
            app.plot_fresh(self.main_surface.triangles[0].t)


    def randomise_parameters(self):
        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                edge.ea.set(round(abs(np.random.random()*100),1))
                edge.eb.set(round(abs(np.random.random()*100),1))
                triangle.triangle_parameter.set(round(abs(np.random.random()*100),1))

                while not app.string_fraction_to_float(edge.ea.get()) > 0:
                    edge.ea.set(round(abs(np.random.random()*100),1))
                while not app.string_fraction_to_float(edge.eb.get()) > 0:
                    edge.eb.set(round(abs(np.random.random()*100),1))
                while not app.string_fraction_to_float(triangle.triangle_parameter.get()) > 0:
                    triangle.triangle_parameter.set(round(abs(np.random.random()*100),1))
        


    def convert_gluing_table_to_surface(self,filename):
        gluing_table = pd.read_table(filename)
        columns = gluing_table.columns.tolist()
        
        columns = columns[0].rsplit(',')
        if 'Triangle Parameter' in columns:
            gluing_table_array = np.array(gluing_table)
            gluing_table_array = np.array([[app.string_fraction_to_float(x) for x in row[0].rsplit(',')[4:]] for row in gluing_table_array])
            for row in gluing_table_array:
                for param in row:
                    assert param > 0
            gluing_table_array_last_col = gluing_table_array[:,-1].copy()
            gluing_table_array[:,-1] = gluing_table_array[:,-2]
            gluing_table_array[:,-2] = gluing_table_array_last_col
            self.input_parameters = gluing_table_array
        edges = columns[1:4]
        gluing_table = np.array(gluing_table.values.tolist())
        self.abstract_surface = AbstractSurface()
        new_gluing_table = []
        for triangle in gluing_table:
            new_gluing_table.append(triangle[0].rsplit(','))
            self.abstract_surface.add_triangle()
        gluing_table = np.array(new_gluing_table)
        for triangle_row in gluing_table:
            triangle = self.abstract_surface.triangles[int(triangle_row[0])]
            triangle_row = triangle_row[1:]
            for edge_index in range(len(edges)):
                current_edge = edges[edge_index].rsplit(' ')[1]

                for edge in triangle.edges:
                    if current_edge == edge.index:
                        current_edge = edge
                        break
                try:
                    [other_triangle_index, other_edge_index] = triangle_row[edge_index].rsplit(' ')
                except:
                    continue
                other_edge_index = other_edge_index[1:-1]
                other_triangle = self.abstract_surface.triangles[int(other_triangle_index)]

                if int(other_edge_index[0]) < int(other_edge_index[1]):
                    for other_edge in other_triangle.edges:
                        if other_edge.index == other_edge_index:
                            self.abstract_surface.glue_edges(current_edge, other_edge, current_edge.v0, other_edge.v0)
                else:
                    for other_edge in other_triangle.edges:
                        if other_edge.index == other_edge_index[::-1]:
                            self.abstract_surface.glue_edges(current_edge, other_edge,current_edge.v0, other_edge.v1)

    def generate_developing_map(self):

        try:
            assert int(self.depth_string.get()) >=0
            self.error_text.set("")
        except:
            self.error_text.set("Please enter a valid non-negative integer value for depth.")
            return

        coord0 = self.abstract_plotting_surface.triangles[0].vertices[0].coord
        coord1 = self.abstract_plotting_surface.triangles[0].vertices[1].coord
        coord2 = self.abstract_plotting_surface.triangles[0].vertices[2].coord
        coord0 = [coord0[0],coord0[1],1]
        coord1 = [coord1[0],coord1[1],1]
        coord2 = [coord2[0], coord2[1], 1]

        self.abstract_surface.orientation = np.sign(np.linalg.det(np.array([coord0, coord1, coord2])))



        for plotting_triangle in self.abstract_plotting_surface.triangles:
            abstract_triangle = self.abstract_surface.triangles[plotting_triangle.index]
            abstract_triangle.triangle_parameter = app.string_fraction_to_float(plotting_triangle.triangle_parameter.get())
            edge_index = 0
            for edge in plotting_triangle.edges:
                flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                abstract_triangle.edges[edge_index].ea = app.string_fraction_to_float(edge.ea.get())
                abstract_triangle.edges[edge_index].eb = app.string_fraction_to_float(edge.eb.get())
                if not flipped:
                    abstract_triangle.edges[edge_index].edge_glued[2].ea = app.string_fraction_to_float(edge.ea.get())
                    abstract_triangle.edges[edge_index].edge_glued[2].eb = app.string_fraction_to_float(edge.eb.get())
                else:
                    abstract_triangle.edges[edge_index].edge_glued[2].ea = app.string_fraction_to_float(edge.eb.get())
                    abstract_triangle.edges[edge_index].edge_glued[2].eb = app.string_fraction_to_float(edge.ea.get())
                edge_index+=1

        self.generate_real_surface_map()
        app.main_surface = self.main_surface
        app.abstract_surface = self.abstract_surface
        app.plot_fresh(self.main_surface.triangles[0].t)
        self.win.destroy()

    def triangle_order_generator(self,edge_list, prev_state, n, top_bottom_list):
        if len(edge_list) == n:
            return (edge_list,top_bottom_list)
        current_edge_on_previous_triangle = edge_list[-1]
        current_edge_on_new_triangle = current_edge_on_previous_triangle.edge_glued[2]
        new_triangle = current_edge_on_new_triangle.triangle
        flipped = 0
        if current_edge_on_previous_triangle.edge_glued[1] != current_edge_on_new_triangle.v0:
            flipped = 1
        for edge_index in range(3):
            if new_triangle.edges[edge_index] == current_edge_on_new_triangle:
                current_edge_on_new_triangle = edge_index
                break
        new_prev_state = 'top'
        if prev_state == 'top':
            next_edge_on_new_triangle = (current_edge_on_new_triangle + ((-1)**flipped)*1) % 3
            if next_edge_on_new_triangle != (current_edge_on_new_triangle + 1) % 3:
                new_prev_state = 'bottom'
        else:
            next_edge_on_new_triangle = (current_edge_on_new_triangle - ((-1) ** flipped) * 1) % 3
            if next_edge_on_new_triangle != (current_edge_on_new_triangle + 1) % 3:
                new_prev_state = 'bottom'

        top_bottom_list.append(new_prev_state)

        next_edge_on_new_triangle = new_triangle.edges[next_edge_on_new_triangle]
        edge_list.append(next_edge_on_new_triangle)
        return self.triangle_order_generator(edge_list, new_prev_state, n, top_bottom_list)

    def generate_new_triangle(self,current_edge, current_abstract_edge, distance_from_initial_triangle, e03, e30, e23, e32, A023, max_distance):


        if distance_from_initial_triangle > max_distance:
            return

        v0, v1, flipped = app.correct_edge_orientation(current_edge)

        r3, c3 = compute_all_until_r3c3(v0.r, v1.r, v0.c,
                                              v1.c, e03, e23,  e30, e32, A023)


        r3_clover, c3_clover = compute_all_until_r3c3(v0.r_clover, v1.r_clover, v0.c_clover,
                                              v1.c_clover, e03, e23,  e30, e32, A023)

        new_triangle = self.main_surface.add_triangle(current_edge, v0, v1, Vertex(c3, r3, c3_clover, r3_clover))


        abstract_triangle = current_abstract_edge.edge_glued[2].triangle

        new_triangle.index = abstract_triangle.index

        edge_index = 0
        for index in range(3):
            if abstract_triangle.edges[index] == current_abstract_edge.edge_glued[2]:
                edge_index = index

        next_edge_indices = [1,-1]


        next_surface_index = 0
        for next_surface_edge in new_triangle.edges[1:]:
            next_abstract_edge = abstract_triangle.edges[(edge_index+next_edge_indices[next_surface_index])%3]
            next_surface_index +=1
            edge_glued = next_abstract_edge.edge_glued[2]
            edge_glued_index = 0
            for index in range(3):
                if edge_glued.triangle.edges[index] == edge_glued:
                    edge_glued_index = index

            edge_forward = edge_glued.triangle.edges[(edge_glued_index + 1) % 3]
            edge_backward = edge_glued.triangle.edges[(edge_glued_index - 1) % 3]

            if edge_backward.index == '02':
                e03 = edge_backward.ea
                e30 = edge_backward.eb
            else:
                e03 = edge_backward.eb
                e30 = edge_backward.ea
            if edge_forward.index == '02':
                e32 = edge_forward.ea
                e23 = edge_forward.eb
            else:
                e32 = edge_forward.eb
                e23 = edge_forward.ea
            A023 = edge_glued.triangle.triangle_parameter
            self.generate_new_triangle(next_surface_edge, next_abstract_edge,
                                  distance_from_initial_triangle+1, e03, e30, e23, e32, A023, max_distance)

        return

    def generate_real_surface_map(self):
        initial_triangle_index = 0
        max_distance = int(self.depth_string.get())-1

        initial_abstract_triangle = self.abstract_surface.triangles[0]
        for triangle in self.abstract_surface.triangles:
            if triangle.index == initial_triangle_index:
                initial_abstract_triangle = triangle

        t = initial_abstract_triangle.triangle_parameter
        e01 = initial_abstract_triangle.edges[0].ea
        e02 = initial_abstract_triangle.edges[2].ea
        e10 = initial_abstract_triangle.edges[0].eb
        e12 = initial_abstract_triangle.edges[1].ea
        e20 = initial_abstract_triangle.edges[2].eb
        e21 = initial_abstract_triangle.edges[1].eb

        cube_root_a_coord_t = np.power(t, (1 / 3))
        c0 = [cube_root_a_coord_t, 0, 0]
        c1 = [0, cube_root_a_coord_t, 0]
        c2 = [0, 0, cube_root_a_coord_t]
        r0 = [0, e01 / cube_root_a_coord_t, e02 / cube_root_a_coord_t]
        r1 = [e10 / cube_root_a_coord_t, 0, e12 / cube_root_a_coord_t]
        r2 = [e20 / cube_root_a_coord_t, e21 / cube_root_a_coord_t, 0]


        c0_clover = [1, 0, 0]
        c1_clover = [0, 1, 0]
        c2_clover = [0, 0, 1]

        x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
        cube_root_x_coord_t = np.power(x_coord_t, 1/3)

        r0_clover = [0, cube_root_x_coord_t, 1]
        r1_clover = [1, 0, cube_root_x_coord_t]
        r2_clover = [cube_root_x_coord_t, 1, 0]




        self.main_surface = Surface(c0, c1, c2, r0, r1, r2, c0_clover, c1_clover , c2_clover, r0_clover, r1_clover, r2_clover)

        self.main_surface.triangles[0].index = initial_abstract_triangle.index
        self.main_surface.triangles[0].t = cube_root_a_coord_t

        for edge_index in range(3):
            edge = initial_abstract_triangle.edges[edge_index]
            edge_glued = initial_abstract_triangle.edges[edge_index].edge_glued[2]
            edge_glued_index = 0
            for index in range(3):
                if edge_glued.triangle.edges[index] == edge_glued:
                    edge_glued_index = index

            edge_forward = edge_glued.triangle.edges[(edge_glued_index+1)%3]
            edge_backward = edge_glued.triangle.edges[(edge_glued_index-1)%3]

            if edge_backward.index == '02':
                e03 = edge_backward.ea
                e30 = edge_backward.eb
            else:
                e03 = edge_backward.eb
                e30 = edge_backward.ea
            if edge_forward.index == '02':
                e32 = edge_forward.ea
                e23 = edge_forward.eb
            else:
                e32 = edge_forward.eb
                e23 = edge_forward.ea

            A023 = edge_glued.triangle.triangle_parameter

            self.generate_new_triangle(self.main_surface.triangles[0].edges[edge_index],  edge, 0, e03, e30, e23, e32, A023, max_distance)

    def get_dual_vertex(self,vertex, edge):
        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
        vertex_is_at_end_of_edge = (edge.v1 == vertex)
        if not flipped:
            if vertex_is_at_end_of_edge:
                other_vertex = edge.edge_glued[2].v1
            else:
                other_vertex = edge.edge_glued[2].v0
        else:
            if vertex_is_at_end_of_edge:
                other_vertex = edge.edge_glued[2].v0
            else:
                other_vertex = edge.edge_glued[2].v1
        return other_vertex



    def find_last_vertex(self,vertex, glued_edge_belonging_to):
        count = 2
        while count == 2:
            vertex = self.get_dual_vertex(vertex, glued_edge_belonging_to)
            for edge in vertex.edges:
                if edge != glued_edge_belonging_to and edge.edge_glued:
                    glued_edge_belonging_to = edge
                    break
            count = 0
            for edge in vertex.edges:
                if edge.edge_glued:
                    count += 1
        return vertex, glued_edge_belonging_to


    def vertex_traversal(self,vertex, vertex_points):

        try:
            self.abstract_plotting_surface.give_vertex_coordinates(vertex,vertex_points.pop())
        except:
            return

        count = 0
        for edge in vertex.edges:
            if edge.edge_glued:
                count+=1

        triangle_belongs_to = vertex.edges[0].triangle
        if count == 0:
            for other_vertex_index in [(vertex.index-1)%3,(vertex.index+1)%3]:
                coord = triangle_belongs_to.vertices[other_vertex_index].coord
                if not len(coord):
                    vertex = triangle_belongs_to.vertices[other_vertex_index]
                    break
        elif count == 1:

            other_vertex_to_consider = None
            non_glued_edge = None
            for edge in vertex.edges:
                if not edge.edge_glued:
                    non_glued_edge = edge
            for other_vertex in [non_glued_edge.v0, non_glued_edge.v1]:
                if other_vertex != vertex:
                    other_vertex_to_consider = other_vertex
            if not len(other_vertex_to_consider.coord):
                vertex = other_vertex_to_consider
            else:
                glued_edge_belonging_to = None
                for edge in vertex.edges:
                    if edge.edge_glued:
                        glued_edge_belonging_to = edge
                other_vertex_to_consider = self.get_dual_vertex(vertex, glued_edge_belonging_to)
                other_count = 0
                for edge in other_vertex_to_consider.edges:
                    if edge.edge_glued:
                        other_count+=1

                if other_count == 1:
                    for other_other_vertex_index in [(other_vertex_to_consider.index - 1) % 3, (other_vertex_to_consider.index + 1) % 3]:
                        coord = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index].coord
                        if not len(coord):
                            vertex = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index]
                            break
                else:
                    other_vertex_to_consider, glued_edge_belonging_to = self.find_last_vertex(vertex, glued_edge_belonging_to)
                    for other_other_vertex_index in [(other_vertex_to_consider.index - 1) % 3, (other_vertex_to_consider.index + 1) % 3]:
                        coord = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index].coord
                        if not len(coord):
                            vertex = other_vertex_to_consider.edges[0].triangle.vertices[other_other_vertex_index]
                            break
        return self.vertex_traversal(vertex, vertex_points)


    def give_edge_identification_color_and_arrow(self):

        colors_ = lambda n: list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF), range(n)))


        edges = []
        for plotting_triangle in self.abstract_plotting_surface.triangles:
            abstract_triangle = self.abstract_surface.triangles[plotting_triangle.index]
            for abstract_edge in abstract_triangle.edges:
                try:
                    abstract_edge.edge_glued[2]
                    flipped = (abstract_edge.edge_glued[1] != abstract_edge.edge_glued[2].v0)
                    edge_to_glue = None
                    for edge in plotting_triangle.edges:
                        if edge.index == abstract_edge.index:
                            edge_to_glue = edge

                    abstract_other_edge = abstract_edge.edge_glued[2]
                    other_edge_to_glue = None
                    for other_triangle in self.abstract_plotting_surface.triangles:
                        for edge in other_triangle.edges:
                            if edge.index == abstract_other_edge.index and other_triangle.index == abstract_other_edge.triangle.index:
                                other_edge_to_glue = edge
                    if not flipped:
                        self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0,
                                                             other_edge_to_glue.v0)
                    else:
                        self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0,
                                                             other_edge_to_glue.v1)
                except:
                    pass

        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                edges.append(edge)

        colors = colors_(2*len(edges)+1)
        arrow_strokes = 1
        for edge in edges:
            new_color = colors.pop()
            edge.color = new_color
            edge.arrow_strokes = arrow_strokes
            try:
                edge.edge_glued[2].color = new_color
                edge.edge_glued[2].arrow_strokes = arrow_strokes
                arrow_strokes += 1
            except:
                edge.color = colors.pop()
                edge.arrow_strokes = 0
        min_stroke = len(edges)+1
        for edge in edges:
            if edge.arrow_strokes > 0:
                min_stroke = min(min_stroke, edge.arrow_strokes)
        for edge in edges:
            if edge.arrow_strokes > 0:
                edge.arrow_strokes = edge.arrow_strokes - min_stroke + 1


    def glue_plotting_surface_edges(self):
        for index in range(len(self.abstract_plotting_surface.triangles[:-1])):
            triangle_plotting_index = self.abstract_plotting_surface.triangles[index].index
            next_triangle_plotting_index = self.abstract_plotting_surface.triangles[index+1].index
            edge_connection_index = '01'
            other_edge_index = '01'
            flipped = 0
            for edge in self.abstract_surface.triangles[triangle_plotting_index].edges:
                try:
                    if edge.edge_glued[2].triangle == self.abstract_surface.triangles[next_triangle_plotting_index]:
                        edge_connection_index = edge.index
                        other_edge_index = edge.edge_glued[2].index
                        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                except:
                    pass
            edge_to_glue = self.abstract_plotting_surface.triangles[index].edges[0]
            other_edge_to_glue = self.abstract_surface.triangles[next_triangle_plotting_index].edges[0]
            for edge in self.abstract_plotting_surface.triangles[index].edges:
                if edge.index == edge_connection_index:
                    edge_to_glue = edge
            for edge in self.abstract_plotting_surface.triangles[index+1].edges:
                if edge.index == other_edge_index:
                    other_edge_to_glue = edge
            if not flipped:
                self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v0)
            else:
                self.abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v1)

    def matplotlib_to_tkinter(self, x):
        x = np.array([[x[0]],[x[1]]])
        mat1 = [-3.031443579456372, -4.828833563156737]
        tk1 = [-66., 87.]
        mat2 = [10.675953475476799, -0.25732417952460196]
        tk2 = [272., 7.]
        M = [[mat1[0], mat1[1], 0, 0], [0, 0, mat1[0], mat1[1]], [mat2[0], mat2[1], 0, 0], [0, 0, mat2[0], mat2[1]]]
        T = [[tk1[0]], [tk1[1]], [tk2[0]], [tk2[1]]]
        M = np.array(M)
        T = np.array(T)
        solution = np.matmul(np.linalg.inv(M), T).T.flatten()
        A = np.array([[solution[0], solution[1]], [solution[2], solution[3]]])
        kt_coords = np.matmul(A, x)
        kt_coords = kt_coords.T.flatten()
        abs_coord_x = self.chart_type.get_tk_widget().winfo_x() + kt_coords[
            0] + 0.5 * self.chart_type.get_tk_widget().winfo_width()
        abs_coord_y = self.chart_type.get_tk_widget().winfo_y() + kt_coords[
            1] + 0.5 * self.chart_type.get_tk_widget().winfo_height()
        return [abs_coord_x, abs_coord_y]

    def generate_parameter_entrance_widgets(self, selected_triangle, event):
        x = self.win.winfo_pointerx()
        y = self.win.winfo_pointery()

        edges = selected_triangle.edges
        try:
            self.parameter_entries[selected_triangle] = []
            [x1, y1] = selected_triangle.vertices[0].coord
            [x2, y2] = selected_triangle.vertices[1].coord
            [x3, y3] = selected_triangle.vertices[2].coord

            centre = [np.mean([x1, x2, x3]), np.mean([y1, y2, y3])]

            [x,y] = self.matplotlib_to_tkinter(centre)

            self.triangle_parameter_entry = ttk.Entry(self.win, textvariable=selected_triangle.triangle_parameter, width=5)
            self.triangle_parameter_entry.place(x=x,y=y)

            for edge_index in range(3):
                    first_coord = np.array(edges[edge_index].v0.coord)
                    second_coord = np.array(edges[edge_index].v1.coord)
                    ea_parameter_string = edges[edge_index].ea
                    eb_parameter_string = edges[edge_index].eb
                    ea_parameter_entry = ttk.Entry(self.win, textvariable=ea_parameter_string, width=5)
                    eb_parameter_entry = ttk.Entry(self.win, textvariable=eb_parameter_string,width=5)
                    [ea_x, ea_y] = self.matplotlib_to_tkinter(2 / 3 * first_coord + 1 / 3 * second_coord)
                    [eb_x, eb_y] = self.matplotlib_to_tkinter(1 / 3 * first_coord + 2 / 3 * second_coord)
                    self.parameter_entries[selected_triangle].append([ea_parameter_entry, eb_parameter_entry])
                    ea_parameter_entry.place(x=ea_x, y=ea_y)
                    eb_parameter_entry.place(x=eb_x, y=eb_y)
        except:
            self.parameter_entries[selected_triangle] = []
            self.parameter_strings[selected_triangle] = []
            [x1, y1] = selected_triangle.vertices[0].coord
            [x2, y2] = selected_triangle.vertices[1].coord
            [x3, y3] = selected_triangle.vertices[2].coord

            centre = [np.mean([x1, x2, x3]), np.mean([y1, y2, y3])]

            [x, y] = self.matplotlib_to_tkinter(centre)
            self.triangle_parameter_string = tk.StringVar(value="1")
            self.triangle_parameter_entry = ttk.Entry(self.win, textvariable=self.triangle_parameter_string, width=5)
            self.triangle_parameter_entry.place(x=x, y=y)
            for edge_index in range(3):
                first_coord = np.array(edges[edge_index].v0.coord)
                second_coord = np.array(edges[edge_index].v1.coord)
                ea_parameter_string = edges[edge_index].ea
                eb_parameter_string = edges[edge_index].eb
                ea_parameter_entry = ttk.Entry(self.win, textvariable=ea_parameter_string,width=5)
                eb_parameter_entry = ttk.Entry(self.win, textvariable=eb_parameter_string,width=5)
                [ea_x, ea_y] = self.matplotlib_to_tkinter(2/3*first_coord+1/3*second_coord)
                [eb_x, eb_y] = self.matplotlib_to_tkinter(1/3*first_coord+2/3*second_coord)
                self.parameter_entries[selected_triangle].append([ea_parameter_entry, eb_parameter_entry])
                self.parameter_strings[selected_triangle].append([ea_parameter_string, eb_parameter_string])
                ea_parameter_entry.place(x=ea_x,y=ea_y)
                eb_parameter_entry.place(x=eb_x, y=eb_y)


    def submit_triangle_params(self, selected_triangle):

        try:
            assert app.string_fraction_to_float(selected_triangle.triangle_parameter.get()) > 0
            for edge in selected_triangle.edges:
                assert app.string_fraction_to_float(edge.ea.get()) > 0
                assert app.string_fraction_to_float(edge.eb.get()) > 0
        except:
            self.error_text.set("One or more variables are not well-defined.")
            return


        selected_triangle.selected = False


        for edge_data in self.parameter_entries[selected_triangle]:
            edge_data[0].destroy()
            edge_data[1].destroy()
        self.parameter_entries = {}
        self.triangle_parameter_entry.destroy()
        self.triangle_parameter_entry = None

        self.error_text.set("")

        self.submit_entries_button.destroy()

        self.plot_combinatorial_map()

    def select_triangle_combinatorial_map(self,event):

        for triangle in self.abstract_plotting_surface.triangles:
            if triangle.selected:
                return


        coord = np.array([event.xdata, event.ydata])
        if not coord[0] or not coord[1]:
            return
        centres = []
        for triangle in self.abstract_plotting_surface.triangles:
            [x1, y1] = triangle.vertices[0].coord
            [x2, y2] = triangle.vertices[1].coord
            [x3, y3] = triangle.vertices[2].coord
            x = [x1, x2, x3]
            y = [y1, y2, y3]
            centres.append([np.mean(x), np.mean(y)])

        centres = np.array(centres)
        distances = np.linalg.norm(np.repeat(np.array([coord]), len(centres), axis=0) - centres, axis=1)
        selected_triangle = self.abstract_plotting_surface.triangles[np.argmin(distances)]


        selected_triangle.selected = True

        self.plot_combinatorial_map()

        self.generate_parameter_entrance_widgets(selected_triangle, event)

        self.submit_entries_button = ttk.Button(self.win, text="Submit Triangle Parameters",
                                          command=lambda:self.submit_triangle_params(selected_triangle))
        self.submit_entries_button.pack(side='right', padx=10, pady=5)




    def plot_combinatorial_map(self):
        self.ax.clear()
        self.ax.remove()
        self.ax.set_axis_off()

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Combinatorial Map')
        self.ax.set_axis_off()
        self.figure.canvas.mpl_connect('button_press_event',self.select_triangle_combinatorial_map)
        plotted_edges = []
        for triangle in self.abstract_plotting_surface.triangles:
            for edge in triangle.edges:
                [x1, y1] = edge.v0.coord
                [x2, y2] = edge.v1.coord
                x = [x1, x2]
                y = [y1, y2]
                self.ax.plot(x, y, c=edge.color)
                plotted_edges.append(edge)
                if edge.arrow_strokes > 0:
                    try:
                        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                        for i in range(edge.arrow_strokes):
                            if flipped and edge.edge_glued[2] in plotted_edges:
                                [x1, y1] = edge.v1.coord
                                [x2, y2] = edge.v0.coord
                            self.ax.arrow(x1, y1, (i + 4) * (x2 - x1) / (edge.arrow_strokes + 7),
                                     (i + 4) * (y2 - y1) / (edge.arrow_strokes + 7), head_width=0.3, color=edge.color)
                    except:
                        pass

        for triangle in self.abstract_plotting_surface.triangles:
            [x1, y1] = triangle.vertices[0].coord
            [x2, y2] = triangle.vertices[1].coord
            [x3, y3] = triangle.vertices[2].coord
            x = [x1, x2, x3, x1]
            y = [y1, y2, y3, y1]
            if triangle.selected:
                self.ax.fill(x,y, "b", alpha=0.2)
            self.ax.annotate(triangle.index, [np.mean(x[:-1]), np.mean(y[:-1])])
            coord0 = np.array([x1, y1])
            coord1 = np.array([x2, y2])
            coord2 = np.array([x3, y3])
            self.ax.annotate(0, 9 * coord0 / 10 + 1 / 10 * (coord1 + coord2), color='grey')
            self.ax.annotate(1, 9 * coord1 / 10 + 1 / 10 * (coord0 + coord2), color='grey')
            self.ax.annotate(2, 9 * coord2 / 10 + 1 / 10 * (coord1 + coord0), color='grey')
        



        self.chart_type.draw()



    def generate_combinatorial_map(self):
        punctured_triangles = []
        for triangle in self.abstract_surface.triangles:
            for edge in triangle.edges:
                if not edge.edge_glued:
                    punctured_triangles.append(triangle)
                    break
        if len(punctured_triangles):
            first_triangle = punctured_triangles[0]
        else:
            first_triangle = self.abstract_surface.triangles[0]
        first_edge_index = 0
        for edge_index in range(3):
            if not first_triangle.edges[edge_index].edge_glued:
                first_edge_index = (edge_index - 1) % 3
                break
        edge_list = [first_triangle.edges[first_edge_index]]
        edge_list, top_bottom_list = self.triangle_order_generator(edge_list, 'top', len(self.abstract_surface.triangles), ['top'])

        triangle_indices = [edge.triangle.index for edge in edge_list]


        vertex_points = []
        a = 10
        b = 10
        thetas = np.linspace(2/(len(edge_list))*np.pi,2*np.pi+1/(len(edge_list))*np.pi, len(edge_list)+2)
        for theta in thetas:
            vertex_points.append(np.array([a*np.cos(theta),-b*np.sin(theta)]))

        self.abstract_plotting_surface = AbstractSurface()
        for triangle_index in triangle_indices:
            self.abstract_plotting_surface.add_triangle()
            self.abstract_plotting_surface.triangles[-1].index = triangle_index

        self.glue_plotting_surface_edges()

        self.vertex_traversal(self.abstract_plotting_surface.triangles[0].vertices[0], vertex_points)

        orientation_first_triangle = np.linalg.det([[v.coord[0], v.coord[1], 1] for v in self.abstract_plotting_surface.triangles[0].vertices])
        if orientation_first_triangle < 0:
            unique_vertices = []
            for triangle in self.abstract_plotting_surface.triangles:
                for vertex in triangle.vertices:
                    if vertex not in unique_vertices:
                        unique_vertices.append(vertex)
            for vertex in unique_vertices:
                vertex.coord = vertex.coord[::-1]
                
            self.abstract_plotting_surface.triangles = self.abstract_plotting_surface.triangles[::-1]

        self.give_edge_identification_color_and_arrow()

        if len(self.input_parameters):
            for triangle in self.abstract_plotting_surface.triangles:
                pass
                triangle.triangle_parameter = tk.StringVar(value=self.input_parameters[triangle.index,0])
                for edge_index in range(3):
                    edge = triangle.edges[edge_index]
                    if edge.index != '02':
                        edge.ea = tk.StringVar(value=self.input_parameters[triangle.index, edge_index+1])
                    else:
                        edge.eb = tk.StringVar(value=self.input_parameters[triangle.index, edge_index+1])
                    flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                    if not flipped:
                        try:
                            edge.edge_glued[2].ea = edge.ea
                        except:
                            pass
                        try:
                            edge.edge_glued[2].eb = edge.eb
                        except:
                            pass
                    else:
                        try:
                            edge.edge_glued[2].ea = edge.eb
                        except:
                            pass
                        try:
                            edge.edge_glued[2].eb = edge.ea
                        except:
                            pass
                

        else:
            for triangle in self.abstract_plotting_surface.triangles:
                triangle.triangle_parameter = tk.StringVar(value="1")
                for edge in triangle.edges:
                    edge.ea = tk.StringVar(value="1")
                    edge.eb = tk.StringVar(value="1")
                    flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                    if not flipped:
                        edge.edge_glued[2].ea = edge.ea
                        edge.edge_glued[2].eb = edge.eb
                    else:
                        edge.edge_glued[2].ea = edge.eb
                        edge.edge_glued[2].eb = edge.ea

        self.plot_combinatorial_map()




def import_file():
    filename = filedialog.askopenfilename(filetypes=[("Excel files", ".csv")])
    if not filename:
        return
    try:
        gluing_table = pd.read_table(filename)
        columns = gluing_table.columns
        for row in np.array(gluing_table):
            for element in row[0].rsplit(','):
                assert element
        combinatorial_plot_window = CombinatorialImport(tk, filename)
    except:
        win = tk.Toplevel()
        win.wm_title("Gluing Table Invalid")
        l = tk.Label(win, text="There was an error uploading this gluing table. Please ensure you have a valid gluing table before continuing.")
        l.pack(side="top",padx=20, pady=10)
        l2 = tk.Label(win, text="Explicit examples of the required structure are available in the 'example_gluing_tables' folder. Note that every edge on all triangles must be glued.")
        l2.pack(side="top",padx=20,pady=(0,10))
        #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
        cancel = ttk.Button(win, text="Close", command=win.destroy)
        cancel.pack(side='right', padx=25, pady=5)


def convert_surface_to_gluing_table(self):
    #print(app.main_surface)
    pass

def export_file():
    try:
        assert app.abstract_surface
    except:
        win = tk.Toplevel()
        win.wm_title("No Uploaded Surface")
        l = tk.Label(win, text="Please ensure you have uploaded and submitted the parameters of a gluing table before exporting.")
        l.pack(padx=20, pady=10)
        #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
        cancel = ttk.Button(win, text="Close", command=win.destroy)
        cancel.pack(side='right', padx=25, pady=5)
        return

    current_dir = os.getcwd()
    dir_name = filedialog.asksaveasfilename(filetypes=[("Excel files", ".csv")])
    if not dir_name:
        return

    if '.csv' in dir_name:
        dir_name = dir_name[:-4]


    gluing_table_data = []
    
    for triangle in app.abstract_surface.triangles:
        row = [triangle.index]
        for edge in triangle.edges:
            flipped = (edge.edge_glued[1]!=edge.edge_glued[2].v0)
            if not flipped:
                row.append(f"{edge.edge_glued[2].triangle.index} ({edge.edge_glued[2].index})")
            else:
                row.append(f"{edge.edge_glued[2].triangle.index} ({edge.edge_glued[2].index[::-1]})")
        gluing_table_data.append(row)
    
    gluing_table_data = np.array(gluing_table_data)
    gluing_table_data_last_col = gluing_table_data[:,-1].copy()
    gluing_table_data[:,-1] = gluing_table_data[:,-2]
    gluing_table_data[:,-2] = gluing_table_data_last_col

    parameter_table_data = []

    for triangle in app.abstract_surface.triangles:
        row = [triangle.triangle_parameter]
        for edge in triangle.edges:
            if edge.index != '02':
                row.append(edge.ea)
            else:
                row.append(edge.eb)
        parameter_table_data.append(row)
    
    parameter_table_data = np.array(parameter_table_data)
    parameter_table_data_last_col = parameter_table_data[:,-1].copy()
    parameter_table_data[:,-1] = parameter_table_data[:,-2]
    parameter_table_data[:,-2] = parameter_table_data_last_col

    column_names = np.array(['Triangle', 'Edge 01', 'Edge 02', 'Edge 12', 'Triangle Parameter', 'Edge 01', 'Edge 02', 'Edge 12'])
    data = np.hstack([gluing_table_data,parameter_table_data])
    table = pd.DataFrame(data)
    table.columns = column_names
    table.to_csv(f"{dir_name}.csv", index=False)

    

    # f = open(f"{dir_name}.csv", "wb+")

    
    # pickle.dump(app.abstract_surface, f)

    # f.close()

def exit_file():
    exit()

def restart_popup():
    win = tk.Toplevel()
    win.wm_title("Restart Program")
    l = tk.Label(win, text="Are you sure you want to restart the program? Any unsaved data will be lost.")
    l.pack(padx=20,pady=10)
    #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
    cancel = ttk.Button(win, text="Cancel", command=win.destroy)
    cancel.pack(side='right', padx=25, pady=5)
    restart = ttk.Button(win,text="Restart", command=restart_program)
    restart.pack(side='right',padx=10,pady=5)

def exit_popup():
    win = tk.Toplevel()
    win.wm_title("Exit Program")
    l = tk.Label(win, text="Are you sure you want to exit the program? Any unsaved data will be lost.")
    l.pack(padx=20, pady=10)
    #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
    cancel = ttk.Button(win, text="Cancel", command=win.destroy)
    cancel.pack(side='right', padx=25, pady=5)
    restart = ttk.Button(win, text="Exit", command=exit_file)
    restart.pack(side='right', padx=10, pady=5)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


def slr3r():
    try:
        assert app.main_surface
        slr3_window = MSL3R()
    except:
        win = tk.Toplevel()
        win.wm_title("Surface Invalid")
        l = tk.Label(win, text="Please ensure you have a valid surface before applying a matrix from SL(3,ℝ).")
        l.pack(padx=20, pady=10)
        #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
        cancel = ttk.Button(win, text="Close", command=win.destroy)
        cancel.pack(side='right', padx=25, pady=5)

def translatelength():
    #translatelength_window = TranslationLength()
    try:
        assert app.abstract_surface
        translatelength_window = TranslationLength()
    except:
        win = tk.Toplevel()
        win.wm_title("No Uploaded Surface")
        l = tk.Label(win, text="Please ensure you have uploaded and submitted the parameters of a gluing table before computing translation lengths.")
        l.pack(padx=20, pady=10)
        #win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
        cancel = ttk.Button(win, text="Close", command=win.destroy)
        cancel.pack(side='right', padx=25, pady=5)



root = tk.Tk()
root.title('Convex Projective Structure Visualisation Tool')
#root.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
root.geometry("1100x520")
menubar = tk.Menu(root)
app = App(root)
filemenu = tk.Menu(menubar, tearoff=0)
transformmenu = tk.Menu(menubar, tearoff=0)
computemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Import Gluing Table (CSV)", command =import_file )
filemenu.add_command(label="Save Imported Parameters", command =export_file )
filemenu.add_command(label="Restart Program", command=restart_popup)
filemenu.add_command(label="Exit", command=exit_popup)
transformmenu.add_command(label="Apply M From SL(3,ℝ)", command=slr3r)
computemenu.add_command(label="𝒜-Coordinates of Centre in Canonical Cell Decomposition", command=app.compute_centre_cell_decomp)
computemenu.add_command(label="Translation Length and Lengths Spectrum", command=translatelength)
menubar.add_cascade(label="File", menu=filemenu)
menubar.add_cascade(label="Transform", menu=transformmenu)
menubar.add_cascade(label="Compute", menu=computemenu)
root.configure(menu=menubar)
app.mainloop()