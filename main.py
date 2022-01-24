import tkinter as tk
from tkinter import ttk
from triangle_class.abstract_triangle import AbstractSurface
from triangle_class.decorated_triangle import *
from visualise.surface_vis import SurfaceVisual
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helper_functions.add_new_triangle_functions import *
from tkinter import filedialog
import pandas as pd
import sys
import os



class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(anchor='nw')
        self.plot_data = []
        self.left_side_frame = ttk.Frame()

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

        self.error_message_label.pack(side='top',anchor='nw', padx=25, pady=0)

        self.add_triangle_frame = ttk.Frame(self.left_side_frame)



        self.add_triangle_param_label = ttk.Label(self.add_triangle_frame, justify='left',
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

        self.normalise_decorations = ttk.Button(self.plot_buttons_frame, text='Normalise Decorations')
        self.normalise_decorations.pack(side='left', anchor='nw', padx=25, pady=0)

        self.generate_surface = ttk.Button(self.plot_buttons_frame, text='Generate Hypersurface')
        self.generate_surface.pack(side='left', anchor='nw', padx=25, pady=0)
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
        self.normalise_decorations.bind('<ButtonPress>',
                                   self.normalise_decorations_function)


    def normalise_decorations_function(self, event):

        try:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title('Cloverleaf Position')
            self.ax.set_axis_off()
            self.edge_selected = self.main_surface.triangles[-1].edges[-2]
            self.main_surface.normalise_vertices()
            for triangle in self.main_surface.triangles:
                [x1, y1, z1] = triangle.vertices[0].c
                [x2, y2, z2] = triangle.vertices[1].c
                [x3, y3, z3] = triangle.vertices[2].c
                [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
                [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
                [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)
                x = [x1, x2, x3, x1]
                y = [y1, y2, y3, y1]

                self.plot_data.append(self.ax.plot(x, y, c='blue'))
            v0 = self.edge_selected.v0.c
            v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
            v1 = self.edge_selected.v1.c
            v1 = clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t)
            self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                                                   [v0[1], v1[1]], c='red'))
            self.chart_type.draw()
            self.generate_surface_error_text.set("")
        except:
            self.generate_surface_error_text.set("Please add an initial triangle before normalising decorations.")



    def onclick(self,event):
        coord = np.array([event.xdata,event.ydata])
        if not coord[0] or not coord[1]:
            return
        all_edges = []
        for triangle in self.main_surface.triangles:
            for edge in triangle.edges:
                if len(edge.triangles)==1:
                    all_edges.append(edge)
        distances = []
        for edge in all_edges:
            v0 = edge.v0.c
            v0 = np.array(clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t))
            v1 = edge.v1.c
            v1 = np.array(clover_position([[v1[0]], [v1[1]], [v1[2]]], self.t))
            v = v1-v0
            x = coord - v0
            distances.append(np.linalg.norm(x - abs(np.dot(x,v))/(np.linalg.norm(v)**2) * v))
        if self.edge_selected:
            self.plot_data[-1][0].remove()
        self.edge_selected = all_edges[np.argmin(distances)]
        v0 = self.edge_selected.v0.c
        v0 = clover_position([[v0[0]],[v0[1]],[v0[2]]], self.t)
        v1 = self.edge_selected.v1.c
        v1 = clover_position([[v1[0]],[v1[1]],[v1[2]]], self.t)
        self.plot_data.append(self.ax.plot([v0[0], v1[0]],
                     [v0[1], v1[1]],c='red'))

        self.chart_type.draw()

    def correct_edge_orientation(self, edge):
        [v0,v1,v2] = edge.triangles[0].vertices
        # if np.linalg.det(np.array([v0.c,v1.c,v2.c])) < 0:
        #     [v0, v1, v2] = [v1, v0, v2]
        vertices = np.array([v0, v1, v2, v0])
        if vertices[np.argwhere(edge.v0 == vertices)[0,0]+1] == edge.v1:
            [edge.v0,edge.v1] = [edge.v1, edge.v0]

    def add_triangle(self, event):
        try:
            assert self.edge_selected
            e03 = float(self.add_triangle_params[0].get())
            e30 = float(self.add_triangle_params[1].get())
            e23 = float(self.add_triangle_params[2].get())
            e32 = float(self.add_triangle_params[3].get())
            A023 = float(self.add_triangle_params[4].get())
            assert e03 > 0 and e30 > 0 and e23 > 0 and e32 > 0 and A023 > 0

            self.correct_edge_orientation(self.edge_selected)
            print(f'r0: {self.edge_selected.v0.r}', f'r2: {self.edge_selected.v1.r}')
            print(f'c0: {self.edge_selected.v0.c}', f'c2: {self.edge_selected.v1.c}')
            m_inverse = compute_m_inverse(self.edge_selected.v0.r, self.edge_selected.v1.r, self.edge_selected.v0.c,
                                              self.edge_selected.v1.c, e03, e23)
            print(np.linalg.det(m_inverse))
            c3 = compute_c3(m_inverse,e03, e23, A023)
            r3 = compute_r3(self.edge_selected.v0.c, self.edge_selected.v1.c, c3, e30, e32)
            print(f'r3: {r3}', f'c3: {c3}')
            self.main_surface.add_triangle(self.edge_selected,Vertex(c3,r3))

            self.add_triangle_error_text.set("")
            if self.edge_selected:
                self.plot_data[-1][0].remove()
            self.edge_selected = self.main_surface.triangles[-1].edges[-2]
            for triangle in self.main_surface.triangles:
                [x1, y1, z1] = triangle.vertices[0].c
                [x2, y2, z2] = triangle.vertices[1].c
                [x3, y3, z3] = triangle.vertices[2].c
                [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
                [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
                [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)
                x = [x1, x2, x3, x1]
                y = [y1, y2, y3, y1]
                self.plot_data.append(self.ax.plot(x, y,c='blue'))

            

            v0 = self.edge_selected.v0.c
            v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
            v1 = self.edge_selected.v1.c
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
            t = float(self.triangle_parameter.get())
            self.t = t
            e01 = float(self.half_edge_params[0].get())
            e10 = float(self.half_edge_params[1].get())
            e02 = float(self.half_edge_params[2].get())
            e20 = float(self.half_edge_params[3].get())
            e12 = float(self.half_edge_params[4].get())
            e21 = float(self.half_edge_params[5].get())
            assert t > 0 and e01 > 0 and e10 > 0 and e02 > 0 and e20 > 0 and e12 > 0 and e21 > 0

            self.error_text.set("")
            self.main_surface = Surface([1,0,0], [0,t,0], [0,0, 1],
                                                             [0, e01/t, e02], [e10, 0, e12], [e20, e21/t, 0])
            self.edge_selected = self.main_surface.triangles[-1].edges[-1]
            for triangle in self.main_surface.triangles:
                [x1, y1, z1] = triangle.vertices[0].c
                [x2, y2, z2] = triangle.vertices[1].c
                [x3, y3, z3] = triangle.vertices[2].c
                [x1, y1] = clover_position([[x1], [y1], [z1]], self.t)
                [x2, y2] = clover_position([[x2], [y2], [z2]], self.t)
                [x3, y3] = clover_position([[x3], [y3], [z3]], self.t)

                x = [x1, x2, x3, x1]
                y = [y1, y2, y3, y1]

                self.plot_data.append(self.ax.plot(x, y,c='blue'))
            v0 = self.edge_selected.v0.c
            v0 = clover_position([[v0[0]], [v0[1]], [v0[2]]], self.t)
            v1 = self.edge_selected.v1.c
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

def convert_surface_to_gluing_table():
    #print(app.main_surface)
    pass

def convert_gluing_table_to_surface(filename):
    gluing_table = pd.read_table(filename)
    columns = gluing_table.columns.tolist()
    columns = columns[0].rsplit(',')
    edges = columns[1:]
    gluing_table = np.array(gluing_table.values.tolist())
    abstract_surface = AbstractSurface()
    new_gluing_table = []
    for triangle in gluing_table:
        new_gluing_table.append(triangle[0].rsplit(','))
        abstract_surface.add_triangle()
    gluing_table = np.array(new_gluing_table)
    for triangle_row in gluing_table:
        triangle = abstract_surface.triangles[int(triangle_row[0])]
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
            other_triangle = abstract_surface.triangles[int(other_triangle_index)]

            if int(other_edge_index[0]) < int(other_edge_index[1]):
                for other_edge in other_triangle.edges:
                    if other_edge.index == other_edge_index:
                        abstract_surface.glue_edges(current_edge, other_edge, current_edge.v0, other_edge.v0)
            else:
                for other_edge in other_triangle.edges:
                    if other_edge.index == other_edge_index[::-1]:
                        abstract_surface.glue_edges(current_edge, other_edge,current_edge.v0, other_edge.v1)

    return abstract_surface
#
# def clover_position(x):
#     x = np.array(x)
#     #P = np.array([[1,-1/2,-1/2],[-1/2,1/2, 0],[-1/2, 0, 1/2]])
#     v = 1 / 3 * np.array([[1], [1], [1]])
#     #x = np.matmul(P,x)
#     T_inverse = np.array([[0, -np.sqrt(2), -1/np.sqrt(2)], [0,0, np.sqrt(3/2)],[-1,1,1]])
#
#     [x,y,z] = np.matmul(T_inverse,x-v)
#     return [-x[0],y[0]]

def clover_position(x,t):
    #P = np.array([[1,-1/2,-1/2],[-1/2,1/2, 0],[-1/2, 0, 1/2]])
    #v = 1 / 3 * np.array([[1], [1], [1]])
    #x = np.matmul(P,x-v)+v
    cube_root1 = np.array([1,0])
    cube_root2 = np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    cube_root3 = np.array([np.cos(-2*np.pi/3), np.sin(-2*np.pi/3)])
    [x,y] = x[0]*cube_root1 + x[1]*cube_root2/t + x[2]*cube_root3
    return [x,y]


def generate_developing_map(abstract_surface):
    pass

def triangle_order_generator(edge_list, prev_state, n, top_bottom_list):
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
    return triangle_order_generator(edge_list, new_prev_state, n, top_bottom_list)

def generate_real_surface_map(abstract_surface,ax):
    punctured_triangles = []
    for triangle in abstract_surface.triangles:
        for edge in triangle.edges:
            if not edge.edge_glued:
                punctured_triangles.append(triangle)
                break
    number_of_outer_edges = len(abstract_surface.triangles)+2
    if len(punctured_triangles):
        first_triangle = punctured_triangles[0]
    else:
        first_triangle = abstract_surface.triangles[0]
    first_edge_index = 0
    for edge_index in range(3):
        if not first_triangle.edges[edge_index].edge_glued:
            first_edge_index = (edge_index - 1) % 3
            break
    edge_list = [first_triangle.edges[first_edge_index]]
    edge_list, top_bottom_list = triangle_order_generator(edge_list, 'top', len(abstract_surface.triangles), ['top'])
    t = 1
    e01 = 1
    e02 = 1
    e10 = 1
    e12 = 1
    e20 = 1
    e21 = 1
    e03 = 1
    e30 = 1
    e23 = 1
    e32 = 1
    A023 = 1
    main_surface = Surface([1,0,0], [0,t,0], [0,0, 1],
                                                             [0, e01/t, e02], [e10, 0, e12], [e20, e21/t, 0])
    current_edge_real_surface = main_surface.triangles[0].edges[0]
    current_real_triangle = main_surface.triangles[0]
    current_abstract_triangle = edge_list[1].triangle
    current_edge_abstract_surface = edge_list[0].edge_glued[2]
    main_surface.triangles[0].index = edge_list[0].triangle.index
    for edge in edge_list[1:]:
        app.correct_edge_orientation(current_edge_real_surface)
        m_inverse = compute_m_inverse(current_edge_real_surface.v0.r, current_edge_real_surface.v1.r, current_edge_real_surface.v0.c,
                                      current_edge_real_surface.v1.c, e03, e23)
        c3 = compute_c3(m_inverse, e03, e23, A023)
        r3 = compute_r3(current_edge_real_surface.v0.c, current_edge_real_surface.v1.c, c3, e30, e32)
        main_surface.add_triangle(current_edge_real_surface, Vertex(c3, r3))
        current_real_triangle = main_surface.triangles[-1]
        current_real_triangle.index = current_abstract_triangle.index

        current_edge_abstract_index = 0
        next_edge_is_anticlockwise = False
        for temp_index in range(3):
            if current_abstract_triangle.edges[temp_index] == current_edge_abstract_surface:
                current_edge_abstract_index = temp_index
                break
        if current_abstract_triangle.edges[(current_edge_abstract_index + 1)%3] == edge:
            next_edge_is_anticlockwise = True
        current_edge_real_surface = current_real_triangle.edges[((-1)**next_edge_is_anticlockwise) % 3]
        current_abstract_triangle = edge.edge_glued[2].triangle
        current_edge_abstract_surface = edge.edge_glued[2]
    #main_surface.normalise_vertices()

    # print(top_bottom_list)
    # print([edge.triangle.index for edge in edge_list])
    # vertex_points = []
    # r=2
    # thetas = np.linspace(0,2*np.pi, len(main_surface.triangles)+2)
    # for theta in thetas:
    #     vertex_points.append(np.array([r*np.cos(theta),r*np.sin(theta)]))
    #
    # first_connecting_edge = main_surface.triangles[0].edges[0]
    # for edge in main_surface.triangles[0].edges:
    #     if len(edge.triangles)==2:
    #         first_connecting_edge = edge
    # for vertex in main_surface.triangles[0].vertices:
    #     if vertex != first_connecting_edge.v0 and vertex != first_connecting_edge.v1:
    #         vertex.c = vertex_points[0]
    #         for other_vertex_index in range(3):
    #             if main_surface.triangles[0].vertices[other_vertex_index] == vertex:
    #                 other_vertex_index = (other_vertex_index -1)%3
    #                 main_surface.triangles[0].vertices[other_vertex_index].c = vertex_points[-1]
    #
    #
    # vertex_points_used = [vertex_points[-1],vertex_points[0]]
    #
    #
    # triangle_index = 1
    #
    #
    #
    # connecting_edges = []
    # for triangle in main_surface.triangles[1:]:
    #     for edge in triangle.edges:
    #         if edge not in connecting_edges and len(edge.triangles) == 2:
    #             if top_bottom_list[triangle_index] == 'top':
    #                 vertex_point_to_use = vertex_points[(-1)**triangle_index*triangle_index]
    #                 edge.v1.c = vertex_point_to_use
    #                 vertex_points_used.append(vertex_point_to_use)
    #             else:
    #                 vertex_point_to_use = vertex_points[-(-1) ** triangle_index * triangle_index]
    #                 edge.v0.c = vertex_point_to_use
    #                 vertex_points_used.append(vertex_point_to_use)
    #             connecting_edges.append(edge)
    #
    #     triangle_index+=1
    #
    #
    # for triangle in main_surface.triangles:
    #     for edge in triangle.edges:
    #         if len(edge.v0.c) == 3:
    #             print(triangle.index)
    #             for v in vertex_points_used:
    #                 for other_v_index in range(len(vertex_points)):
    #                     if np.all(v!=vertex_points[other_v_index]):
    #                         edge.v0.c = v




    for triangle in main_surface.triangles:
        [x1, y1, z1] = triangle.vertices[0].c
        [x2, y2, z2] = triangle.vertices[1].c
        [x3, y3, z3] = triangle.vertices[2].c
        # try:
        #     [x3, y3] = triangle.vertices[2].c
        # except:
        #     print(triangle.index)
        [x1,y1] = clover_position([[x1],[y1],[z1]], t)
        [x2, y2] = clover_position([[x2],[y2],[z2]], t)
        [x3, y3] = clover_position([[x3],[y3],[z3]], t)
        x = [x1, x2, x3, x1]
        y = [y1, y2, y3, y1]
        ax.plot(x, y)
        ax.annotate(triangle.index,[np.mean(x[:-1]),np.mean(y[:-1])])

    return main_surface

def generate_combinatorial_map(abstract_surface, ax):
    punctured_triangles = []
    for triangle in abstract_surface.triangles:
        for edge in triangle.edges:
            if not edge.edge_glued:
                punctured_triangles.append(triangle)
                break
    number_of_outer_edges = len(abstract_surface.triangles) + 2
    if len(punctured_triangles):
        first_triangle = punctured_triangles[0]
    else:
        first_triangle = abstract_surface.triangles[0]
    first_edge_index = 0
    for edge_index in range(3):
        if not first_triangle.edges[edge_index].edge_glued:
            first_edge_index = (edge_index - 1) % 3
            break
    edge_list = [first_triangle.edges[first_edge_index]]
    edge_list, top_bottom_list = triangle_order_generator(edge_list, 'top', len(abstract_surface.triangles), ['top'])

    triangle_indices = [edge.triangle.index for edge in edge_list]
    print(top_bottom_list)

    vertex_points = []
    r=10
    thetas = np.linspace(0.5*np.pi,2*np.pi, len(edge_list)+2)
    for theta in thetas:
        vertex_points.append(np.array([-r*np.cos(theta),r*np.sin(theta)]))

    abstract_plotting_surface = AbstractSurface()
    for triangle_index in triangle_indices:
        abstract_plotting_surface.add_triangle()
        abstract_plotting_surface.triangles[-1].index = triangle_index

    for index in range(len(abstract_plotting_surface.triangles[:-1])):
        triangle_plotting_index = abstract_plotting_surface.triangles[index].index
        next_triangle_plotting_index = abstract_plotting_surface.triangles[index+1].index
        edge_connection_index = '01'
        other_edge_index = '01'
        flipped = 0
        for edge in abstract_surface.triangles[triangle_plotting_index].edges:
            try:
                if edge.edge_glued[2].triangle == abstract_surface.triangles[next_triangle_plotting_index]:
                    edge_connection_index = edge.index
                    other_edge_index = edge.edge_glued[2].index
                    flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
            except:
                pass
        edge_to_glue = abstract_plotting_surface.triangles[index].edges[0]
        other_edge_to_glue = abstract_surface.triangles[next_triangle_plotting_index].edges[0]
        for edge in abstract_plotting_surface.triangles[index].edges:
            if edge.index == edge_connection_index:
                edge_to_glue = edge
        for edge in abstract_plotting_surface.triangles[index+1].edges:
            if edge.index == other_edge_index:
                other_edge_to_glue = edge
        if not flipped:
            abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v0)
        else:
            abstract_plotting_surface.glue_edges(edge_to_glue, other_edge_to_glue, edge_to_glue.v0, other_edge_to_glue.v1)

    print([triangle.index for triangle in abstract_plotting_surface.triangles])
    non_glued_vertex = 0

    # for triangle in abstract_plotting_surface.triangles:
    #
    #     for edge in triangle.edges:
    #         print('triangle', triangle.index, 'edge:', edge.index, edge.edge_glued)


    for vertex in abstract_plotting_surface.triangles[0].vertices:
        belongs_to_glued_edge = False
        for edge in vertex.edges:
            if edge.edge_glued:
                belongs_to_glued_edge = True
        if not belongs_to_glued_edge:
            #print([None != edge.edge_glued for edge in vertex.edges])
            abstract_plotting_surface.give_vertex_coordinates(vertex, vertex_points[0])
            abstract_plotting_surface.give_vertex_coordinates(vertex.edges[0].triangle.vertices[(vertex.index-1) % 3], vertex_points[-1])

    used_vertex_points = [vertex_points[0], vertex_points[-1]]

    for triangle_index in range(len(abstract_plotting_surface.triangles)):
        triangle = abstract_plotting_surface.triangles[triangle_index]
        for edge in triangle.edges:
            if edge.edge_glued:
                try:
                    edge.was_connected
                except:
                    next_vertex = edge.v0
                    for other_edge in edge.v0.edges:
                        try:
                            other_edge.was_connected
                            next_vertex = edge.v1
                        except:
                            pass
                    if top_bottom_list[triangle_index] == 'top':
                        next_vertex_point = vertex_points[triangle_index+1]
                    else:
                         next_vertex_point = vertex_points[-triangle_index -2]
                    abstract_plotting_surface.give_vertex_coordinates(next_vertex, next_vertex_point)
                    used_vertex_points.append(next_vertex_point)

                    edge.was_connected = True
                    edge.edge_glued[2].was_connected = True

    last_vertex = abstract_plotting_surface.triangles[-1].vertices[0]
    for vertex in abstract_plotting_surface.triangles[-1].vertices:
        try:
            vertex.coord
        except:
            last_vertex = vertex

    for other_coord_index in range(len(vertex_points)):
        was_used = False
        for coord in used_vertex_points:
            if coord[0] == vertex_points[other_coord_index][0] and coord[1] == vertex_points[other_coord_index][1]:
                was_used = True
                break
        if not was_used:
            last_vertex.coord = vertex_points[other_coord_index]


    for triangle in abstract_plotting_surface.triangles:
        for vertex in triangle.vertices:
                try:
                    print('triangle: ', triangle.index,'vertex: ', vertex.index, 'coords: ',vertex.coord)
                except:
                    print('triangle: ', triangle.index, 'vertex: ', vertex.index, 'coords: ', 'None')

    for triangle in abstract_plotting_surface.triangles:
        [x1, y1] = triangle.vertices[0].coord
        [x2, y2] = triangle.vertices[1].coord
        [x3, y3] = triangle.vertices[2].coord
        x = [x1, x2, x3, x1]
        y = [y1, y2, y3, y1]
        ax.plot(x, y)
        ax.annotate(triangle.index, [np.mean(x[:-1]), np.mean(y[:-1])])


def import_file():
    filename = filedialog.askopenfilename(filetypes=[("Excel files", ".csv")])
    if not filename:
        return
    abstract_surface = convert_gluing_table_to_surface(filename)
    win = tk.Toplevel()
    win.wm_title("Uploaded Surface")
    l = tk.Label(win, text="The uploaded gluing table is visualised as a combinatorial map below. Continue importing?")
    l.pack(padx=20, pady=10)
    win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
    figure = plt.Figure(figsize=(6, 5), dpi=100)
    ax = figure.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure, win)
    chart_type.get_tk_widget().pack()
    ax.set_title('Combinatorial Map')

    generate_combinatorial_map(abstract_surface, ax)
    #main_surface = generate_real_surface_map(abstract_surface,ax)
    ax.set_axis_off()
    chart_type.draw()




    cancel = ttk.Button(win, text="Cancel", command=win.destroy)
    cancel.pack(side='right', padx=25, pady=5)
    continue_button= ttk.Button(win, text="Continue", command=lambda : (generate_developing_map(abstract_surface), win.destroy()) )
    continue_button.pack(side='right', padx=10, pady=5)

def export_file():
    try:
        app.main_surface
    except:
        win = tk.Toplevel()
        win.wm_title("Surface Invalid")
        l = tk.Label(win, text="Please ensure you have a valid surface before exporting.")
        l.pack(padx=20, pady=10)
        win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
        cancel = ttk.Button(win, text="Close", command=win.destroy)
        cancel.pack(side='right', padx=25, pady=5)
        return

    current_dir = os.getcwd()
    dir_name = filedialog.askdirectory()  # asks user to choose a directory
    if not dir_name:
        return
    os.chdir(dir_name)  # changes your current directory
    gluing_data = convert_surface_to_gluing_table()

    os.chdir(current_dir)

def exit_file():
    exit()

def restart_popup():
    win = tk.Toplevel()
    win.wm_title("Restart Program")
    l = tk.Label(win, text="Are you sure you want to restart the program? Any unsaved data will be lost.")
    l.pack(padx=20,pady=10)
    win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
    cancel = ttk.Button(win, text="Cancel", command=win.destroy)
    cancel.pack(side='right', padx=25, pady=5)
    restart = ttk.Button(win,text="Restart", command=restart_program)
    restart.pack(side='right',padx=10,pady=5)

def exit_popup():
    win = tk.Toplevel()
    win.wm_title("Exit Program")
    l = tk.Label(win, text="Are you sure you want to exit the program? Any unsaved data will be lost.")
    l.pack(padx=20, pady=10)
    win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
    cancel = ttk.Button(win, text="Cancel", command=win.destroy)
    cancel.pack(side='right', padx=25, pady=5)
    restart = ttk.Button(win, text="Exit", command=exit_file)
    restart.pack(side='right', padx=10, pady=5)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


root = tk.Tk()
root.title('Convex Projective Surface Visualisation Tool')
root.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
root.geometry("1000x520")
menubar = tk.Menu(root)
app = App(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Import Gluing Table (CSV)", command =import_file )
filemenu.add_command(label="Export Gluing Table (CSV)", command =export_file )
filemenu.add_command(label="Restart Program", command=restart_popup)
filemenu.add_command(label="Exit", command=exit_popup)
menubar.add_cascade(label="File", menu=filemenu)
root.configure(menu=menubar)
app.mainloop()