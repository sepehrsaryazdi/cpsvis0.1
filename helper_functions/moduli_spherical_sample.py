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


class ModuliSphericalSample():
    def __init__(self, max_r=100,n=10, theta_n=50, tree_depth=2):
        
        self.win = tk.Toplevel()
        self.win.wm_title("Minimum Length Spectrum Over Moduli Space (Spherical)")
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
        self.controls_label = tk.Label(self.controls_frame, text="Use the sliders to control the value of θᵢ for i ∈ {1,2,3,4,5,6,7}.\nTick the corresponding box if you would like to sweep this angle for θ ∈ [0,2π] during plot generation.")
        self.controls_label.pack()
        
        self.slider_frames = []
        
        for i in range(7):
            self.slider_frames.append(tk.Frame(self.controls_frame))
        self.angle_labels = [] 

        for i in range(1,8):
            self.angle_labels.append(tk.Label(self.slider_frames[i-1],text=f'θ{integer_to_script(i,up=False)}'))

        self.value_string_vars = []
        for i in range(7):
            new_text_variable = tk.StringVar()
            self.value_string_vars.append(new_text_variable)

        self.sliders = []
        for i in range(7):
            if i != 6:
                self.sliders.append(ttk.Scale(self.slider_frames[i], from_=0, to=np.pi, orient="horizontal", command=self.update_display, length=300))
            else:
                self.sliders.append(ttk.Scale(self.slider_frames[i], from_=0, to=2*np.pi,orient="horizontal", command=self.update_display,  length=300))
        
        for slider in self.sliders[:-1]:
            slider.set(np.pi/4)
        
        self.sliders[-1].set(np.pi/2)

        self.value_labels = [] 
        for i in range(7):
            self.value_labels.append(tk.Label(self.slider_frames[i],textvariable=self.value_string_vars[i]))

        self.sweep_states = []
        for i in range(7):
            self.sweep_states.append(tk.IntVar())


        self.sweep_checkboxes = []

        for i in range(7):
            
            self.sweep_checkboxes.append(ttk.Checkbutton(self.slider_frames[i], variable=self.sweep_states[i], command=self.update_selections_function_generator(i)))
            
        #self.lambda_functions[2]()


        i = 0
        for slider in self.sliders:
            self.angle_labels[i].pack(side='left')
            slider.pack(side='left')
            self.value_labels[i].pack(side='left')
            self.sweep_checkboxes[i].pack(side='left',padx=5)
            
            i+=1
        
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

        self.tree_depth_variable = tk.StringVar(value=self.tree_depth)
        self.angle_sweeps_variable = tk.StringVar(value=self.theta_n)
        self.radius_samples_variable = tk.StringVar(value=self.n)
        self.max_r_variable = tk.StringVar(value=self.max_r)

        self.parameters = [self.tree_depth_variable, self.angle_sweeps_variable, self.max_r_variable,self.radius_samples_variable]
        texts = ["Length Tree Depth", "Number of Angle Sweep Samples","Maximum R Value", "Number of Radius Samples"]
        
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
            
        self.theta_n = int(string_fraction_to_float(self.angle_sweeps_variable.get()))
        self.tree_depth = int(string_fraction_to_float(self.tree_depth_variable.get()))
        self.n = int(string_fraction_to_float(self.radius_samples_variable.get()))
        self.max_r = int(string_fraction_to_float(self.max_r_variable.get()))

    

        index_sweep = -1
        i = 0
        for sweep_variable in self.sweep_states:
            if sweep_variable.get() == 1:
                index_sweep = i
            i+=1
        
        self.index_sweep = index_sweep
        


        self.generate_minimum_lengths()

    
    def update_selections_function_generator(self,index_value):
        
        def update_selections():
            for i in range(7):
                if i != index_value:
                    self.sweep_states[i].set(0)
        return update_selections
        
    
    def update_display(self,e):
        
        for i in range(7):
            self.value_string_vars[i].set(f'{self.display_coefficient(self.sliders[i].get())}')

    
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
        equations = ["$A = 1+R\cos(\\theta_1)$\n",
                    "$B = 1+R\sin(\\theta_1)\cos(\\theta_2)$\n",
                    "$a^- = 1 + R\sin(\\theta_1)\sin(\\theta_2)\cos(\\theta_3)$\n",
                    "$a^+ = 1 + R\sin(\\theta_1)\sin(\\theta_2)\sin(\\theta_3)\cos(\\theta_4)$\n",
                    "$b^- = 1 + R\sin(\\theta_1)\sin(\\theta_2)\sin(\\theta_3)\sin(\\theta_4)\cos(\\theta_5)$\n",
                    "$b^+ = 1 + R\sin(\\theta_1)\sin(\\theta_2)\sin(\\theta_3)\sin(\\theta_4)\sin(\\theta_5)\cos(\\theta_6)$\n",
                    "$e^- = 1 + R\sin(\\theta_1)\sin(\\theta_2)\sin(\\theta_3)\sin(\\theta_4)\sin(\\theta_5)\sin(\\theta_6)\cos(\\theta_7)$\n",
                    "$e^+ = 1 + R\sin(\\theta_1)\sin(\\theta_2)\sin(\\theta_3)\sin(\\theta_4)\sin(\\theta_5)\sin(\\theta_6)\sin(\\theta_7)$\n"]



        self.equations_ax.text(0, 0.4, ''.join(equations))
        self.equations_ax.text(0,0.2, '$0\leq \\theta_1,\\theta_2,\\theta_3,\\theta_4,\\theta_5,\\theta_6 \leq \pi$\n$0\leq \\theta_7\leq 2\pi$')
        





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
        
        theta_n = self.theta_n
        thetas = np.array([theta_value.get() for theta_value in self.sliders])
        
        if self.index_sweep !=-1:
            radiis = []
            theta_space = np.linspace(np.pi/theta_n,2*np.pi,theta_n)
            minimum_lengths_r_theta = []
            for i in range(len(thetas)-1):
                if thetas[i] == 0:
                    thetas[i]+=0.0001
                elif thetas[i] == np.pi:
                    thetas[i]-=0.0001
            
            if thetas[-1] == 0:
                thetas[-1] +=0.0001
            elif thetas[-1] == np.pi*2:
                thetas[-1]=0.0001
            
            
            for theta in theta_space:
                
                self.update_progress_bar(theta/max(theta_space))
                thetas[self.index_sweep] = theta
                [radii, coordinates] = self.get_all_x_coordinates(thetas)
                minimum_lengths = self.generate_minimum_length_distribution(coordinates)
                radiis.append(radii)
                minimum_lengths_r_theta.append(minimum_lengths)
            
            self.figure = plt.figure(figsize=(7,5))
            
            self.ax = self.figure.add_subplot(1,1,1,projection='3d')

            all_lengths = []
            for minimum_lengths in minimum_lengths_r_theta:
                for length in minimum_lengths:
                    all_lengths.append(length)
            
            max_height = max(all_lengths)
            min_height = min(all_lengths)
            norm = mpl.colors.Normalize(vmin=min_height, vmax=max_height)
            cmap = cm.jet
            m = cm.ScalarMappable(norm=norm, cmap=cmap)

            


            for theta_index in range(len(theta_space)):
                radii = radiis[theta_index]
                theta = theta_space[theta_index]
                minimum_lengths = minimum_lengths_r_theta[theta_index]
                N = len(radii)
                X = radii*np.cos(theta)
                Y = radii*np.sin(theta)
                Z = minimum_lengths
                for i in range(N-1):
                    self.ax.plot(X[i:i+2],Y[i:i+2],Z[i:i+2],color=plt.cm.jet((1/2*(Z[i]+Z[i+1])-min_height)/(max_height-min_height)))
                #self.ax.plot3D(radii*np.cos(theta), radii*np.sin(theta), minimum_lengths, c=minimum_lengths)
            
            self.ax.set_xlabel(f'$R\\cos(\\theta_{self.index_sweep+1})$')
            self.ax.set_ylabel(f'$R\\sin(\\theta_{self.index_sweep+1})$')
            self.ax.set_zlabel(f'Minimum Length')

            theta_values = [f"\\theta_{i+1} = {self.display_coefficient(thetas[i])}" for i in range(len(thetas))]
            theta_values[self.index_sweep] = f'\\theta_{self.index_sweep+1} ∈ [0,2π]'

            theta_string = f"${', '.join(theta_values)}$"

            coordinate_latex = r"$\mathcal{A}$-coordinates" if "𝒜" in self.coordinate_variable.get() else r"$\mathcal{X}$-coordinates"

            self.ax.set_title(f"Minimum Lengths against $R$ for $\\theta_{self.index_sweep+1} \in [0,2\pi]$ ({coordinate_latex})\n{theta_string}")
            
            m._A = []
            clb = self.figure.colorbar(m)
            clb.set_label('Minimum Length')
            self.figure.canvas.manager.set_window_title('Minimum Lengths Spectrum Over Moduli Space Plot')
            self.update_progress_bar(0)
            
            self.figure.show()

        else:
            self.theta_n = 1
            [radii, coordinates] = self.get_all_x_coordinates(thetas)
            minimum_lengths = self.generate_minimum_length_distribution(coordinates)
            self.figure = plt.figure(figsize=(7,5))
            self.ax = self.figure.add_subplot(1,1,1)
            self.ax.plot(radii, minimum_lengths)
            self.ax.set_xlabel("$R$ (Distance from 𝟙)")
            self.ax.set_ylabel("Minimum Length")

            theta_values = [f"\\theta_{i+1} = {self.display_coefficient(thetas[i])}" for i in range(len(thetas))]

            theta_string = f"${', '.join(theta_values)}$"


            coordinate_latex = r"$\mathcal{A}$-coordinates" if "𝒜" in self.coordinate_variable.get() else r"$\mathcal{X}$-coordinates"


            self.ax.set_title(f"Minimum Lengths against $R$ ({coordinate_latex})\n{theta_string}")
            self.figure.canvas.manager.set_window_title('Minimum Lengths Spectrum Over Moduli Space Plot')
            self.update_progress_bar(0)
            self.figure.show()




        
    
    def generate_minimum_length_distribution(self,coordinates):
        minimum_lengths = []
        i=0
        for coordinate in coordinates:
            #if self.theta_n == 1:
            #self.update_progress_bar(self.progress_var.get()/100 + i/(self.theta_n*len(coordinates)))
            min_length = self.get_min_length_from_x(coordinate)
            minimum_lengths.append(min_length)
            i+=1
        return np.array(minimum_lengths)

    def get_min_length_from_x(self,x):
        if "𝒜" in self.coordinate_variable.get():
            x = a_to_x_coordinate_torus(x)
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
        precision_halfs = 10
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
        







#ModuliSample(100,10)