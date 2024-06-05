import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os
import io

# Get the current directory path
current_directory = os.getcwd()

def select_image():
    filepath = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        original_image = Image.open(filepath)
        original_image.thumbnail((200, 200))  # Resize the image to fit the label
        original_photo = ImageTk.PhotoImage(original_image)
        original_label.config(image=original_photo)
        original_label.image = original_photo  # Keep a reference to the image

        # Pass the image to the processing script
        python_path = os.path.join(current_directory, ".venv", "Scripts", "python.exe")
        process = subprocess.Popen([python_path, "image_processing.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=filepath.encode())

        # Display the flipped image
        flipped_image = Image.open(io.BytesIO(stdout))
        flipped_image.thumbnail((200, 200))  # Resize the image to fit the label
        flipped_photo = ImageTk.PhotoImage(flipped_image)
        flipped_label.config(image=flipped_photo)
        flipped_label.image = flipped_photo  # Keep a reference to the image

# Create the Tkinter GUI
root = tk.Tk()
root.title("Image Viewer")

# Button for selecting an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Label to display the original image
original_label = tk.Label(root)
original_label.pack(side=tk.LEFT)

# Label to display the flipped image
flipped_label = tk.Label(root)
flipped_label.pack(side=tk.RIGHT)

root.mainloop()
