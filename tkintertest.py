import tkinter as tk
from tkinter import ttk
from decorated_triangles.triangle import *
from visualise.surface_vis import SurfaceVisual
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(anchor='nw')

        self.left_side_frame = ttk.Frame()

        self.create_surface_frame = ttk.Frame(self.left_side_frame)
        self.triangle_parameter_label = ttk.Label(self.create_surface_frame,justify='left', text='Set Initial Triangle Parameter (Must be non-zero)')
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

        self.generate_button = ttk.Button(self.surface_buttons_frame,text='Generate Surface')
        self.generate_button.pack(side='left',anchor='nw', padx=20, pady=25)
        self.surface_buttons_frame.pack(side='top', anchor='nw')

        self.error_text = tk.StringVar()
        self.error_text.set("")
        self.error_message_label = tk.Label(self.left_side_frame,justify='left',textvariable=self.error_text, fg='red')

        self.error_message_label.pack(side='top',anchor='nw', padx=25, pady=0)
        self.left_side_frame.pack(side='left',anchor='nw')
        # Create the application variable.
        self.triangle_parameter = tk.StringVar()
        self.half_edge_params = []
        for half_edge_param_entry in self.half_edge_param_entries:
            self.half_edge_params.append(tk.StringVar())
            self.half_edge_params[-1].set(1)
            half_edge_param_entry["textvariable"] = self.half_edge_params[-1]
        # Set it to some value.
        self.triangle_parameter.set("1")
        # Tell the entry widget to watch this variable.
        self.entry_parameter["textvariable"] = self.triangle_parameter

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(self.figure, root)
        chart_type.get_tk_widget().pack()
        self.ax.set_title('Projected Combinatorial Map (A-coordinates)')

        # Define a callback for when the user hits return.
        # It prints the current value of the variable.
        self.generate_button.bind('<ButtonPress>',
                             self.generate_surface)
        self.randomise_button.bind('<ButtonPress>',
                            self.randomise_numbers)

    def randomise_numbers(self,event):
        for half_edge_param in self.half_edge_params:
            half_edge_param.set(round(abs(np.random.random()*10),1))
        param = round(-10+np.random.random()*20,1)
        while param == 0:
            param = round(-10+np.random.random()*20,1)
        self.triangle_parameter.set(param)


    def generate_surface(self, event):

        try:
            t = float(self.triangle_parameter.get())
            e01 = float(self.half_edge_params[0].get())
            e10 = float(self.half_edge_params[1].get())
            e02 = float(self.half_edge_params[2].get())
            e20 = float(self.half_edge_params[3].get())
            e12 = float(self.half_edge_params[4].get())
            e21 = float(self.half_edge_params[5].get())
            assert e01 > 0 and e10 > 0 and e02 > 0 and e20 > 0 and e12 > 0 and e21 > 0

            self.error_text.set("")
            main_surface = Surface(2, 1, Triangle(Decoration([1,0,0], [0,t,0], [0,0, 1],
                                                         [0, e01/t, e02], [e10, 0, e12], [e20, e21/t, 0])))
            surface_vis = SurfaceVisual(main_surface)
            surface_vis.show_vis()
        except:
            self.error_text.set("One or more variables are not well-defined.")

root = tk.Tk()
root.title('Convex Projective Surface Visualisation Tool')
root.tk.call("source", "./misc/azure.tcl")
root.iconphoto(False, tk.PhotoImage(file='./misc/Calabi-Yau.png'))
root.geometry("1000x520")
app = App(root)
app.mainloop()