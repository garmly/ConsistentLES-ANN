import tkinter as tk

global Lx,Ly,Lz,Nx,Ny,Nz,dx,dy,dz,U0,nu,sample_index

root = tk.Tk()
root.title("PyDNS")

name_label = tk.Label(root, text="Name:")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack()

root.mainloop()
