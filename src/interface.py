import tkinter as tk
import numpy as np
from tkinter.font import Font

def submit_form():
    # Function to handle form submission
    # Add your logic here to process the form data
    global Lx,Ly,Lz,Nx,Ny,Nz,U0,nu,write_interval
    Lx = float(entries[0].get())
    Ly = float(entries[1].get())
    Lz = float(entries[2].get())
    Nx = int(entries[3].get())
    Ny = int(entries[4].get())
    Nz = int(entries[5].get())
    U0 = float(entries[6].get())
    nu = float(entries[7].get())
    write_interval = float(entries[8].get())
    root.destroy()

root = tk.Tk()
root.geometry(f"250x600")
root.resizable(False, False)
root.title("PyDNS v0.0")

# Setting up fonts
title_font = Font(family="Helvetica", size=24, weight="bold")
heading_font = Font(family="Helvetica", size=12)

# Title
title_label = tk.Label(root, text="pyDNS", font=title_font)
title_label.pack(pady=20)

# Heading 1
h1_label = tk.Label(root, text="Grid parameters", font=heading_font)
h1_label.pack(anchor="n",pady=10)

# Create the form labels and entries
labels = ["Lx:","Ly:","Lz:",
        "Nx:","Ny:","Nz:",]
entries = []

for label_text in labels:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text=label_text, width=10, anchor="w")
    label.pack(side="left")
    entry = tk.Entry(frame, relief="groove", bd=2)
    entry.pack(side="right")
    entries.append(entry)

labels = ["U0","Viscosity"]

# Heading 2
h2_label = tk.Label(root, text="Fluid parameters", font=heading_font)
h2_label.pack(anchor="n",pady=10)

for label_text in labels:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text=label_text, width=10, anchor="w")
    label.pack(side="left")
    entry = tk.Entry(frame, relief="groove", bd=2)
    entry.pack(side="right")
    entries.append(entry)

# Heading 3
h2_label = tk.Label(root, text="Output parameters", font=heading_font)
h2_label.pack(anchor="n",pady=10)

# Write interval
frame = tk.Frame(root)
frame.pack(pady=5)
label = tk.Label(frame, text="Write interval", width=10, anchor="w")
label.pack(side="left")
entry = tk.Entry(frame, relief="groove", bd=2)
entry.pack(side="right")
entries.append(entry)

# submit button
submit_button = tk.Button(root, text="Start", command=submit_form)
submit_button.pack(pady=20)

root.mainloop()
