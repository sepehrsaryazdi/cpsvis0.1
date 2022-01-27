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


def clover_position(x, t):
    x = np.array(x)
    x = x/(sum(sum(x)))
    #P = np.array([[1,-1/2,-1/2],[-1/2,1/2, 0],[-1/2, 0, 1/2]])
    v = 1 / 3 * np.array([[1], [1], [1]])
    #x = np.matmul(P,x)
    T_inverse = np.array([[0, -np.sqrt(2), -1/np.sqrt(2)], [0,0, np.sqrt(3/2)],[-1,1,1]])

    [x,y,z] = np.matmul(T_inverse,x-v)
    return [x[0],y[0]]


# def clover_position(x,t):
#     #P = np.array([[1,-1/2,-1/2],[-1/2,1/2, 0],[-1/2, 0, 1/2]])
#     #v = 1 / 3 * np.array([[1], [1], [1]])
#     #x = np.matmul(P,x-v)+v
#     cube_root1 = np.array([1,0])
#     cube_root2 = np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
#     cube_root3 = np.array([np.cos(-2*np.pi/3), np.sin(-2*np.pi/3)])
#     [x,y] = x[0]*cube_root1 + x[1]*cube_root2/t + x[2]*cube_root3
#     return [x,y]

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


    def normalise_decorations_function(self, event):

        try:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.remove()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title('Cloverleaf Position')
            self.ax.set_axis_off()
            self.edge_selected = self.main_surface.triangles[-1].edges[-2]
            self.main_surface.normalise_vertices()
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
        except:
            self.generate_surface_error_text.set("Please add an initial triangle before normalising decorations.")

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
                if len(edge.triangles)==1:
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
        [v0,v1,v2] = edge.triangles[0].vertices
        # if np.linalg.det(np.array([v0.c,v1.c,v2.c])) < 0:
        #     [v0, v1, v2] = [v1, v0, v2]
        vertices = np.array([v0, v1, v2, v0])
        if vertices[np.argwhere(edge.v0 == vertices)[0,0]+1] == edge.v1:
            [edge.v0,edge.v1] = [edge.v1, edge.v0]

    def add_triangle(self, event):
        try:
            assert self.edge_selected
            e03 = self.string_fraction_to_float(self.add_triangle_params[0].get())
            e30 = self.string_fraction_to_float(self.add_triangle_params[1].get())
            e23 = self.string_fraction_to_float(self.add_triangle_params[2].get())
            e32 = self.string_fraction_to_float(self.add_triangle_params[3].get())
            A023 = self.string_fraction_to_float(self.add_triangle_params[4].get())
            assert e03 > 0 and e30 > 0 and e23 > 0 and e32 > 0 and A023 > 0

            self.correct_edge_orientation(self.edge_selected)
            # print(f'r0: {self.edge_selected.v0.r}', f'r2: {self.edge_selected.v1.r}')
            # print(f'c0: {self.edge_selected.v0.c}', f'c2: {self.edge_selected.v1.c}')

            r3, c3 = compute_all_until_r3c3(self.edge_selected.v0.r, self.edge_selected.v1.r, self.edge_selected.v0.c,
                                              self.edge_selected.v1.c, e03, e23, e30, e32, A023)

            r3_clover, c3_clover = compute_all_until_r3c3(self.edge_selected.v0.r_clover, self.edge_selected.v1.r_clover, self.edge_selected.v0.c_clover,
                                              self.edge_selected.v1.c_clover, e03, e23, e30, e32, A023)
            # print(f'r0: {self.edge_selected.v0.r}', f'r2: {self.edge_selected.v1.r}')
            # print(f'c0: {self.edge_selected.v0.c}', f'c2: {self.edge_selected.v1.c}')
            # print(f'm_inverse: {m_inverse}')
            # print(f'r3: {r3}', f'c3: {c3}')
            print(f'r0_clover: {self.edge_selected.v0.r_clover}', f'r2_clover: {self.edge_selected.v1.r_clover}')
            print(f'c0_clover: {self.edge_selected.v0.c_clover}', f'c2_clover: {self.edge_selected.v1.c_clover}')
            #print(f'm_inverse: {m_inverse_clover}')
            print(f'r3_clover: {r3_clover}', f'c3_clover: {c3_clover}')
            self.main_surface.add_triangle(self.edge_selected,Vertex(c3,r3, c3_clover, r3_clover))

            self.add_triangle_error_text.set("")
            if self.edge_selected:
                self.plot_data[-1][0].remove()
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
            #self.generate_surface_visual(None)
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
            c0_clover = [1,0,0]
            c1_clover = [0,1,0]
            c2_clover = [0,0,1]
            x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
            cube_root_x_coord_t = np.power(x_coord_t, 1)
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
            # first_line0 = np.array([[0], [0], [1]]) - 3 * np.array([[0], [1], [0]])
            # first_line1 = np.array([[0], [0], [1]]) + 3 * np.array([[0], [1], [0]])
            # second_line0 = np.array([[1], [0], [0]]) - 3 * np.array([[0], [0], [1]])
            # second_line1 = np.array([[1], [0], [0]]) + 3 * np.array([[0], [0], [1]])
            # third_line0 = np.array([[0], [1], [0]]) - 3 * np.array([[1], [0], [0]])
            # third_line1 = np.array([[0], [1], [0]]) + 3 * np.array([[1], [0], [0]])
            # for l in [[first_line0, first_line1], [second_line0, second_line1], [third_line0, third_line1]]:
            #     [x1, y1] = clover_position(l[0], self.t)
            #     [x2, y2] = clover_position(l[1], self.t)
            #     x = [x1, x2]
            #     y = [y1, y2]
            #     self.plot_data.append(self.ax.plot(x, y, c='blue'))
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
    def __init__(self, tk, filename):
        self.tk = tk
        self.parameter_entries = {}
        self.parameter_strings = {}
        self.triangle_parameter_string = None
        self.triangle_parameter_entry = None
        self.convert_gluing_table_to_surface(filename)
        self.win = self.tk.Toplevel()
        self.win.resizable(width=False, height=False)
        self.win.wm_title("Uploaded Surface")
        self.l = tk.Label(self.win,
                     text="The uploaded gluing table is visualised as a combinatorial map below. You can change any edge and triangle parameters by selecting a triangle. Once you're done, press continue.")
        self.l.pack(padx=20, pady=10)
        self.win.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
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
                                     command=lambda: (self.generate_developing_map(), self.win.destroy()))
        self.continue_button.pack(side='right', padx=10, pady=5)






    def convert_gluing_table_to_surface(self,filename):
        gluing_table = pd.read_table(filename)
        columns = gluing_table.columns.tolist()
        columns = columns[0].rsplit(',')
        edges = columns[1:]
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
        self.generate_real_surface_map()
        app.main_surface = self.main_surface
        app.plot_fresh(self.main_surface.triangles[0].t)

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

        app.correct_edge_orientation(current_edge)

        # m_inverse = compute_m_inverse(current_edge.v0.r, current_edge.v1.r, current_edge.v0.c,
        #                                       current_edge.v1.c, e03, e23)
        # c3 = compute_c3(m_inverse, e03, e23, A023)
        # r3 = compute_r3(current_edge.v0.c, current_edge.v1.c, c3, e30, e32)

        r3, c3 = compute_all_until_r3c3(current_edge.v0.r, current_edge.v1.r, current_edge.v0.c,
                                              current_edge.v1.c, e03, e23,  e30, e32, A023)

        # m_inverse_clover = compute_m_inverse(current_edge.v0.r_clover, current_edge.v1.r_clover,
        #                                       current_edge.v0.c_clover,
        #                                       current_edge.v1.c_clover, e03, e23)
        # c3_clover = compute_c3(m_inverse_clover, e03, e23, A023)
        # r3_clover = compute_r3(current_edge.v0.c_clover, current_edge.v1.c_clover, c3_clover, e30, e32)

        r3_clover, c3_clover = compute_all_until_r3c3(current_edge.v0.r_clover, current_edge.v1.r_clover, current_edge.v0.c_clover,
                                              current_edge.v1.c_clover, e03, e23,  e30, e32, A023)

        # print('r0: ', current_edge.v0.r_clover, 'r2:', current_edge.v1.r_clover)
        # print('c0: ', current_edge.v0.c_clover, 'c2: ', current_edge.v1.c_clover)
        # print('r3: ', r3_clover, 'c3: ', c3_clover)

        new_triangle = self.main_surface.add_triangle(current_edge, Vertex(c3, r3, c3_clover, r3_clover))

        abstract_triangle = current_abstract_edge.edge_glued[2].triangle

        new_triangle.index = abstract_triangle.index

        abstract_triangle = current_abstract_edge.edge_glued[2].triangle

        flipped = (current_abstract_edge.edge_glued[1] != current_abstract_edge.edge_glued[2].v0)

        edge_index = 0
        for edge_index in range(3):
            if abstract_triangle.edges[edge_index] == current_abstract_edge:
                edge_index = current_abstract_edge.index

        next_surface_index = 0
        for next_surface_edge in new_triangle.edges[1:]:
            if flipped:
                next_abstract_edge = abstract_triangle.edges[(edge_index-(next_surface_index+1))%3]
            else:
                next_abstract_edge =  abstract_triangle.edges[(edge_index + (next_surface_index + 1)) % 3]

            next_surface_index += 1
            self.generate_new_triangle(next_surface_edge, next_abstract_edge,
                                  distance_from_initial_triangle+1, e03, e30, e23, e32, A023, max_distance)
        return

    def generate_real_surface_map(self):
        initial_triangle_index = 0
        max_distance = 5

        initial_abstract_triangle = self.abstract_surface.triangles[0]
        for triangle in self.abstract_surface.triangles:
            if triangle.index == initial_triangle_index:
                initial_abstract_triangle = triangle

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
        c0 = [1,0,0]
        c1 = [0,t,0]
        c2 = [0,0,1]
        r0 = [0, e01/t, e02]
        r1 = [e10, 0, e12]
        r2= [e20, e21/t, 0]
        c0_clover = [1,0,0]
        c1_clover = [0,1,0]
        c2_clover = [0,0,1]
        x_coord_t = compute_t(e01, e12, e20, e10, e21, e02)
        cube_root_x_coord_t = np.power(x_coord_t, 1)
        r0_clover = [0, cube_root_x_coord_t, 1]
        r1_clover = [1, 0, cube_root_x_coord_t]
        r2_clover = [cube_root_x_coord_t, 1, 0]

        self.main_surface = Surface(c0, c1, c2, r0, r1, r2, c0_clover, c1_clover ,c2_clover,r0_clover, r1_clover, r2_clover)

        self.main_surface.triangles[0].index = initial_abstract_triangle.index



        for edge_index in range(3):
            self.generate_new_triangle(self.main_surface.triangles[0].edges[edge_index],  initial_abstract_triangle.edges[edge_index], 0, e03, e30, e23, e32, A023, max_distance)

        #print(main_surface.triangles)


    def get_dual_vertex(self,vertex, edge):
        flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
        #print(edge.edge_glued[1].index, edge.edge_glued[2].v0.index)
        vertex_is_at_end_of_edge = (edge.v1 == vertex)
        #print('vertex_is_at_end_of_edge',vertex_is_at_end_of_edge)
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

        number_of_unique_colours = int(3*len(self.abstract_plotting_surface.triangles)/2)
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
        #print(event.x, event.y)
        x = self.win.winfo_pointerx()
        y = self.win.winfo_pointery()

        click_tk_coords = [self.win.winfo_pointerx() - self.win.winfo_rootx(), self.win.winfo_pointery() - self.win.winfo_rooty()]
        #print(self.win.winfo_pointerx() - self.win.winfo_rootx(), self.win.winfo_pointery() - self.win.winfo_rooty())



        click_mat_coords = [event.xdata, event.ydata]

        #print(event.xdata, event.ydata)


        desired_x = 0
        desired_y = 0
        # abs_coord_x = self.win.winfo_pointerx() - self.win.winfo_rootx() +desired_x
        # abs_coord_y = self.win.winfo_pointery() - self.win.winfo_rooty() + desired_y

        origin_tk_coords = [self.chart_type.get_tk_widget().winfo_x() + 0.5*self.chart_type.get_tk_widget().winfo_width(), self.chart_type.get_tk_widget().winfo_y() + 0.5*self.chart_type.get_tk_widget().winfo_height()]





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



        #self.parameter_entry_point = ttk.Entry(self.win)
        #[abs_coord_x, abs_coord_y]= self.matplotlib_to_tkinter([event.xdata, event.ydata])

        #self.parameter_entry_point.place(x=abs_coord_x, y=abs_coord_y)


    def submit_triangle_params(self, selected_triangle):
        for triangle in self.abstract_plotting_surface.triangles:
            triangle.selected = False

        for edge_data in self.parameter_entries[selected_triangle]:
            edge_data[0].destroy()
            edge_data[1].destroy()
        self.parameter_entries = {}
        self.triangle_parameter_entry.destroy()
        self.triangle_parameter_entry = None

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

        #print([triangle.index for triangle in abstract_plotting_surface.triangles])

        self.vertex_traversal(self.abstract_plotting_surface.triangles[0].vertices[0], vertex_points)
        self.give_edge_identification_color_and_arrow()

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

        # for triangle in abstract_plotting_surface.triangles:
        #     [x1, y1] = triangle.vertices[0].coord
        #     [x2, y2] = triangle.vertices[1].coord
        #     [x3, y3] = triangle.vertices[2].coord
        #     x = [x1, x2, x3, x1]
        #     y = [y1, y2, y3, y1]
        #     ax.plot(x, y)
        #     ax.annotate(triangle.index, [np.mean(x[:-1]), np.mean(y[:-1])])

        self.plot_combinatorial_map()




def import_file():
    filename = filedialog.askopenfilename(filetypes=[("Excel files", ".csv")])
    if not filename:
        return
    combinatorial_plot_window = CombinatorialImport(tk, filename)

def convert_surface_to_gluing_table(self):
    #print(app.main_surface)
    pass

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