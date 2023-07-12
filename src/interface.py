import tkinter as tk
from tkinter.font import Font

def submit_form():
    # Function to handle form submission
    # Add your logic here to process the form data
    global Lx,Ly,Lz,Nx,Ny,Nz,Nxf,Nyf,Nzf,U0,nu,write_interval,max_time,read,filename

    #read = entries[13].variable.get()
    read = read.get()
    filename = entries[14].get()
    U0 = float(entries[9].get())
    nu = float(entries[10].get())
    write_interval = int(entries[11].get())
    max_time = float(entries[12].get())
    Nxf = int(entries[6].get())
    Nyf = int(entries[7].get())
    Nzf = int(entries[8].get())

    if not read:
        Lx = float(entries[0].get())
        Ly = float(entries[1].get())
        Lz = float(entries[2].get())
        Nx = int(entries[3].get())
        Ny = int(entries[4].get())
        Nz = int(entries[5].get())

    root.destroy()

root = tk.Tk()
root.geometry(f"250x500")
root.resizable(True, True)
root.title("PyDNS v0.0")

# Create a scrollbar
scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Setting up fonts
title_font = Font(family="Helvetica", size=24, weight="bold")
heading_font = Font(family="Helvetica", size=12)

# Title
title_label = tk.Label(root, text="pyDNS", font=title_font)
title_label.pack(pady=5)

# Create a frame to hold the form
form_frame = tk.Frame(root)
form_frame.pack(pady=10)

# Create a canvas to hold the form frame and scrollbar
canvas = tk.Canvas(form_frame, yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill="both", expand=True)

# Configure the scrollbar to work with the canvas
scrollbar.config(command=canvas.yview)

# Create a frame inside the canvas to hold the form content
form_content_frame = tk.Frame(canvas)
form_content_frame.pack(fill="both", expand=True)

# Add the form content to the frame
labels = ["Lx:", "Ly:", "Lz:", "Nx:", "Ny:", "Nz:", "Nxf:", "Nyf:", "Nzf:", "U0:", "Viscosity:", "Write interval:", "Max time:", "Read from file?", "Filename:"]
default_value = [1, 1, 1, 32, 32, 32, 16, 16, 16, 1, 1e-6, 1, 10, True, "HIT_256.init"]
entries = []

i = 0
for label_text in labels:
    frame = tk.Frame(form_content_frame)
    frame.pack(pady=5)
    
    label = tk.Label(frame, text=label_text, width=14, anchor="w")
    label.pack(side="left")
    
    if i != 13:
        entry = tk.Entry(frame, relief="groove", bd=2)
        entry.insert(0, default_value[i])
        entry.pack(side="right")
    else:
        read = tk.BooleanVar()
        entry = tk.Checkbutton(frame, variable=read, onvalue=True, offvalue=False)
        entry.pack(side="right")
    
    entries.append(entry)
    i += 1

# Pack the form content frame inside the canvas
canvas.create_window((0, 0), window=form_content_frame, anchor="nw")

# Configure the canvas to scroll with the scrollbar
form_content_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Submit button
submit_button = tk.Button(root, text="Start", command=submit_form)
submit_button.pack(pady=5)

root.mainloop()
