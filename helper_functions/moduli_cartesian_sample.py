from re import I
import string
import numpy as np
from sympy import linsolve, minimum
from helper_functions.add_new_triangle_functions import a_to_x_coordinate_torus, outitude_edge_params, integer_to_script, string_fraction_to_float
from helper_functions.add_new_triangle_functions import compute_translation_matrix_torus
from helper_functions.length_heat_map import LengthHeatMapTree
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


class ModuliCartesianSample():
    def __init__(self, max_r=100,n=10, theta_n=50, tree_depth=4):
        
        self.win = tk.Toplevel()
        self.win.wm_title("Minimum Length Spectrum Over Moduli Space (Cartesian)")
        self.l = tk.Label(self.win, text="Use the following configurations to produce the minimum length spectrum over a slice in the moduli space.")
        self.l.pack()        
        self.triangle_figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.triangle_ax = self.triangle_figure.add_subplot(111)
        self.visual_frame = tk.Frame(self.win)
        self.triangle_chart_type = FigureCanvasTkAgg(self.triangle_figure, self.visual_frame)
        self.triangle_ax.set_axis_off()
        self.plot_triangle()
        self.triangle_chart_type.get_tk_widget().pack(side='right')
        self.equations_figure = plt.Figure(figsize=(5, 5), dpi=100)
        self.equations_ax = self.equations_figure.add_subplot(111)
        self.equations_chart_type = FigureCanvasTkAgg(self.equations_figure, self.visual_frame)
        self.equations_ax.set_axis_off()
        self.plot_equations()
        
        self.equations_chart_type.get_tk_widget().pack(side='right')
        self.visual_frame.pack()
        self.bottom_frame = tk.Frame(self.win)
        self.controls_frame = tk.Frame(self.bottom_frame)
        self.controls_label = tk.Label(self.controls_frame, text="Use the buttons to control the value of vᵢ for i ∈ {1,2,3,4,5,6,7,8}.\n Tick the corresponding box on the left to give -1 or the box on the right for +1.")
        self.controls_label.pack()
        
        self.slider_frames = []
        
        for i in range(8):
            self.slider_frames.append(tk.Frame(self.controls_frame))
        self.angle_labels = [] 

        for i in range(1,9):
            self.angle_labels.append(tk.Label(self.slider_frames[i-1],text=f'v{integer_to_script(i,up=False)}'))

        self.value_string_vars = []
        for i in range(8):
            new_text_variable = tk.StringVar()
            new_text_variable.set("0")
            self.value_string_vars.append(new_text_variable)

        
        self.value_labels = [] 
        for i in range(8):
            self.value_labels.append(tk.Label(self.slider_frames[i],textvariable=self.value_string_vars[i]))

        self.plus_states = []
        self.neg_states = []
        for i in range(8):
            self.plus_states.append(tk.IntVar())
            self.neg_states.append(tk.IntVar())


        self.plus_checkboxes = []
        self.neg_checkboxes = []

        for i in range(8):
            
            self.plus_checkboxes.append(ttk.Checkbutton(self.slider_frames[i], variable=self.plus_states[i], command=self.update_selections_function_generator(i, 1)))
            self.neg_checkboxes.append(ttk.Checkbutton(self.slider_frames[i], variable=self.neg_states[i], command=self.update_selections_function_generator(i, -1)))
            
        #self.lambda_functions[2]()


        for i in range(len(self.angle_labels)):
            self.neg_checkboxes[i].pack(side='left',padx=5)
            self.angle_labels[i].pack(side='left')
            self.plus_checkboxes[i].pack(side='left',padx=5)
            self.value_labels[i].pack(side='left')
            
        
        for slider_frame in self.slider_frames:
            slider_frame.pack()
        self.controls_frame.pack(side='left')

        self.sep = ttk.Separator(self.bottom_frame,orient='vertical')
        self.sep.pack(fill='y',side='left',padx=25)

        self.parameter_frame = tk.Frame(self.bottom_frame)

        self.progress_var = tk.DoubleVar()

        self.progress = ttk.Progressbar(self.parameter_frame, orient='horizontal', length=100,variable=self.progress_var,maximum=100)
        

        self.parameter_label = tk.Label(self.parameter_frame, text="Configure the parameters of plotting below.")
        self.parameter_label.pack(pady=5)

        self.index_sweep = -1
        self.theta_n = theta_n
        self.tree_depth = tree_depth
        self.n = n
        self.max_r = max_r
        self.k = 2

        self.tree_depth_variable = tk.StringVar(value=self.tree_depth)
        self.radius_samples_variable = tk.StringVar(value=self.n)
        self.max_r_variable = tk.StringVar(value=max_r)
        self.k_variable = tk.StringVar(value=self.k)

        self.parameters = [self.tree_depth_variable, self.max_r_variable ,self.radius_samples_variable, self.k_variable]
        texts = ["Length Tree Depth", "Maximum R Value", "Number of Radius Samples", "Max Order of Minima"]
        
        self.parameter_frames = []
        for i in range(len(texts)):
            self.parameter_frames.append(tk.Frame(self.parameter_frame))

        self.parameter_texts = []
        self.parameter_entries = []
        i=0
        for parameter in self.parameters:
            self.parameter_texts.append(tk.Label(self.parameter_frames[i],text=texts[i]))
            self.parameter_entries.append(ttk.Entry(self.parameter_frames[i], textvariable=parameter, width=5))
            self.parameter_texts[-1].pack(side='left')
            self.parameter_entries[-1].pack(side='right')
            i+=1
        
        

        


        
        
        self.parameter_frames.append(tk.Frame(self.parameter_frame))
        
        self.coordinate_variable = tk.StringVar()
        self.coordinate_variable.set("𝒜-coordinates")
        self.coordinate_text=  tk.Label(self.parameter_frames[-1], text="Coordinates: ")
        self.coordinate_text.pack(side='left')
        self.toggle_coordinates = ttk.OptionMenu(self.parameter_frames[-1], self.coordinate_variable, "𝒜-coordinates", "𝒜-coordinates", "𝒳-coordinates")
        self.toggle_coordinates.pack(side="left")

        for parameter_frame in self.parameter_frames:
            parameter_frame.pack(anchor='w')


        self.error_message_variable = tk.StringVar(value="")
        self.error_message_label = tk.Label(self.parameter_frame, textvariable=self.error_message_variable, fg='red')
        self.error_message_label.pack()
        
        self.parameter_frames.append(tk.Frame(self.parameter_frame))

        
        

        self.generate_plot_button = ttk.Button(self.parameter_frame, text="Generate Plot")
    
        self.generate_plot_button.bind("<ButtonPress>", self.generate_plot_command)


        self.generate_plot_button.pack()
        
        

        self.parameter_frame.pack(side='left',pady=(0,2))
        
        


        self.bottom_frame.pack(pady=5)


        
    
    def update_progress_bar(self,value):
        self.progress_var.set(100*value)
        
        if value > 0:
            self.progress.pack()
        else:
            self.progress.pack_forget()
        
        self.progress.update()
        

        
    def generate_plot_command(self,e):

        for parameter in self.parameters:
            try:
                assert string_fraction_to_float(parameter.get()) == int(string_fraction_to_float(parameter.get())) and string_fraction_to_float(parameter.get()) > 0
                self.error_message_variable.set("")
            except:
                self.error_message_variable.set("One or more parameters are not well-defined.\nPlease ensure they are valid positive integers.")
                return
            
        self.theta_n = 1
        self.tree_depth = int(string_fraction_to_float(self.tree_depth_variable.get()))
        self.n = int(string_fraction_to_float(self.radius_samples_variable.get()))
        self.max_r = int(string_fraction_to_float(self.max_r_variable.get()))
        self.k = int(string_fraction_to_float(self.k_variable.get()))


        self.generate_minimum_lengths()

    
    def update_selections_function_generator(self,index_value, sign):
        
        def update_selections():
            if sign > 0:
                self.neg_states[index_value].set(0)
            elif sign < 0:
                self.plus_states[index_value].set(0)
            
            self.update_display("")
        return update_selections
        
    
    def update_display(self,e):
        
        for i in range(8):
            self.value_string_vars[i].set(-int(self.neg_states[i].get()) + int(self.plus_states[i].get()))

    
    def display_coefficient(self,string):
        string_to_return = 'π'
        string_coefficient_value = round(string/np.pi,3)
        if np.isclose(string_coefficient_value, 1):
            return f'{string_to_return}'
        elif np.isclose(string_coefficient_value, int(string_coefficient_value)):
            return f'{int(string_coefficient_value)}{string_to_return}'
        else:
            return f'{string_coefficient_value}{string_to_return}'


    
    def plot_equations(self):
        equations = ["$A = 1+R v_1$\n",
                    "$B = 1+Rv_2$\n",
                    "$a^- = 1 + Rv_3$\n",
                    "$a^+ = 1 + R v_4$\n",
                    "$b^- = 1 + R v_5$\n",
                    "$b^+ = 1 + R v_6$\n",
                    "$e^- = 1 + R v_7$\n",
                    "$e^+ = 1 + R v_8$\n"]



        self.equations_ax.text(0, 0.4, ''.join(equations))
        self.equations_ax.text(0,0.2, '$v_1,v_2,v_3,v_4,v_5,v_6,v_7,v_8\in \{-1,0,1\}$')
        





    def plot_triangle(self):
        coordsy = np.array([0,-2,0,2,0])
        coordsx=np.array([-2,0,2,0,-2])
        arrow_colors=['red','blue','red','blue']
        for i in range(len(coordsx)-1):
            x0 = coordsx[i]
            y0 = coordsy[i]
            x1 = coordsx[i+1]
            y1 = coordsy[i+1]
            self.triangle_ax.plot([x0,x1],[y0,y1],color=arrow_colors[i])
        #self.triangle_ax.plot(coordsx,coordsy,colors=arrow_colors)
        letters = ['b⁺','b⁻','a⁻','a⁺','b⁻','b⁺','a⁺','a⁻']
        
        for i in range(len(coordsy)-1):
            if i not in [3,0]:
                x0 = 2/3*coordsx[i]+1/3*coordsx[i+1]
                y0 = 2/3*coordsy[i]+1/3*coordsy[i+1]
                dx = 1/2*coordsx[i+1]+1/2*coordsx[i]-x0
                dy = 1/2*coordsy[i+1]+1/2*coordsy[i] - y0
                self.triangle_ax.arrow(x0,y0,dx,dy,head_width=0.1,color=arrow_colors[i])
            else:
                x0 = 1/3*coordsx[i]+2/3*coordsx[i+1]
                y0 = 1/3*coordsy[i]+2/3*coordsy[i+1]
                dx = 1/2*coordsx[i+1]+1/2*coordsx[i]-x0
                dy = 1/2*coordsy[i+1]+1/2*coordsy[i] - y0
                self.triangle_ax.arrow(x0,y0,dx,dy,head_width=0.1,color=arrow_colors[i])
            
            x0 = 2/3*coordsx[i]+1/3*coordsx[i+1]
            y0 = 2/3*coordsy[i]+1/3*coordsy[i+1]
            x1 = 1/3*coordsx[i] + 2/3*coordsx[i+1]
            y1 = 1/3*coordsy[i] + 2/3*coordsy[i+1]
            tangent = [x1-x0,y1-y0]
            normal = [tangent[1],-tangent[0]]
            self.triangle_ax.plot([x0-1/15*normal[0],x0+1/15*normal[0]],[y0-1/15*normal[1],y0+1/15*normal[1]],color=arrow_colors[i])
            self.triangle_ax.plot([x1-1/15*normal[0],x1+1/15*normal[0]],[y1-1/15*normal[1],y1+1/15*normal[1]],color=arrow_colors[i])
            self.triangle_ax.annotate(letters[2*i],np.array([x0,y0])+1/4*np.array(normal))
            self.triangle_ax.annotate(letters[2*i+1], np.array([x1,y1])+1/4*np.array(normal))
        
        middle_letters = ['e⁻','e⁺']
        x0 = 2/3*np.array([0,2])+1/3*np.array([0,-2])
        x1 = 1/3*np.array([0,2])+2/3*np.array([0,-2])
        tangent = x1-x0
        normal = np.array([tangent[1],-tangent[0]])
        self.triangle_ax.plot([0,0],[2,-2],color='green')
        x0left = x0 - 1/15*normal
        x0right = x0 + 1/15*normal
        x1left = x1- 1/15*normal
        x1right = x1+1/15*normal
        
        self.triangle_ax.plot([x0left[0],x0right[0]],[x0left[1],x0right[1]], color='green')
        self.triangle_ax.plot([x1left[0],x1right[0]],[x1left[1],x1right[1]], color='green')
        self.triangle_ax.arrow(x0[0],x0[1],1/2.2*tangent[0],1/2.2*tangent[1], head_width=0.1, color='green')
        self.triangle_ax.annotate(middle_letters[0],x0-1/8*np.array(normal))
        self.triangle_ax.annotate(middle_letters[1],x1-1/8*np.array(normal))

        self.triangle_ax.annotate('A',[1/2+0.2,0.1], color='purple',fontsize=30)
        self.triangle_ax.scatter(1/2+0.2,0,color='purple')

        self.triangle_ax.annotate('B',[-1/2-0.2,0.1], color='darkblue',fontsize=30)
        self.triangle_ax.scatter(-1/2-0.2,0,color='darkblue')

        self.triangle_ax.set_xlim(-2,2)
        #self.triangle_ax.set_ylim(-2.5,2.5)
        



        
    
    def generate_minimum_lengths(self):
        self.theta_n = 1
        v = np.array([-float(self.neg_states[i].get())+float(self.plus_states[i].get()) for i in range(8)])
        [radii, coordinates] = self.get_all_x_coordinates(v)
        minimum_lengths = self.generate_minimum_length_distribution(coordinates)
        #print(minimum_lengths)
        self.figure = plt.figure(figsize=(7,5))
        self.ax = self.figure.add_subplot(1,1,1)
        for i in range(len(minimum_lengths[0,:])):
            self.ax.plot(radii, minimum_lengths[:,i], label=f'{i+1}')
        self.ax.set_xlabel("$R$ (Distance from 𝟙)")
        self.ax.set_ylabel("Minimum Length")
        self.ax.legend(loc='best',title='Minima Order')

        v_values = [f"v_{i+1} = {-int(self.neg_states[i].get())+int(self.plus_states[i].get())}" for i in range(8)]

        v_string = f"${', '.join(v_values)}$"

        coordinate_latex = r"$\mathcal{A}$-coordinates" if "𝒜" in self.coordinate_variable.get() else r"$\mathcal{X}$-coordinates"

        self.ax.set_title(f"Minimum Lengths against $R$ ({coordinate_latex})\n{v_string}")
        self.figure.canvas.manager.set_window_title('Minimum Lengths Spectrum Over Moduli Space Plot')
        self.update_progress_bar(0)
        self.figure.show()




        
    
    def generate_minimum_length_distribution(self,coordinates):
        minimum_lengths = []
        i=0
        for coordinate in coordinates:
            
            #if self.theta_n == 1:
            self.update_progress_bar(i/len(coordinates))
            #self.update_progress_bar(self.progress_var.get()/100 + i/(self.theta_n*len(coordinates)))
            min_lengths = self.get_min_length_from_x(coordinate)
            print(min_lengths)
            minimum_lengths.append(min_lengths)
            
            i+=1
        return np.array(minimum_lengths)

    def get_min_length_from_x(self,x):
        if "𝒜" in self.coordinate_variable.get():
            x = a_to_x_coordinate_torus(x)
        alpha1,alpha2 = compute_translation_matrix_torus(x)
        
        lengthheatmaptree = LengthHeatMapTree(self.tree_depth, 1/2, alpha1,alpha2,k=self.k)
        min_lengths = lengthheatmaptree.k_smallest_lengths
        #print(np.linalg.norm(x-1),)
        return min_lengths
    

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
    def get_single_x_coordinate(self,v,r):
        return 1+r*v
        
    def get_all_x_coordinates(self,v):
        precision_halfs = 20
        number_of_halfs = 0
        number_of_halfs_neg = 0
        original_h = 1
        h = original_h
        r = 0
        coordinates = []
        radii = []
        r_max = self.max_r
        while r < self.max_r:
            x = self.get_single_x_coordinate(v,r)

            if not np.all([xi>0 for xi in x]):
                while not np.all([xi>0 for xi in x]):
                    r-=h
                    x = self.get_single_x_coordinate(v,r)
                h = h/2
                number_of_halfs_neg +=1
                if precision_halfs-1 == number_of_halfs_neg:
                    r_max = r
                    break
                

            if not self.outitudes_positive(x):
                while not self.outitudes_positive(x):
                    r -= h
                    x = self.get_single_x_coordinate(v,r)
                
                h = h/2
                number_of_halfs += 1
                if precision_halfs-1 == number_of_halfs:
                    r_max = r
                    break
            r+=h

        radii = np.linspace(0,r_max,self.n)
        #print(radii)
        coordinates = np.array([self.get_single_x_coordinate(v,r) for r in radii])

        
        return [np.array(radii), np.array(coordinates)]
        







#ModuliSample(100,10)