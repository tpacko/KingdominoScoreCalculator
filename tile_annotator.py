import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import shutil

# Config
IMG_SIZE = 128  # for display, not for saving
TILES_FOLDER = 'tiles'
CODE2TERR = {"f":"forest","me":"meadow","mi":"mine",
             "w":"water","wa":"wasteland","wh":"wheat","c":"castle"}
TILE_CLASSES = list(CODE2TERR.keys())
CROWN_CLASSES = [0, 1, 2, 3]

def get_next_free_name(tile, crowns):
    # Find next available name for this tile/crown
    n = 1
    while True:
        name = f"{tile}{n}"
        if crowns > 0:
            name += f"_c{crowns}"
        fname = name + ".png"
        if not os.path.exists(os.path.join(TILES_FOLDER, fname)):
            return fname
        n += 1

# List files needing labeling (name is only a number)
files = [f for f in os.listdir(TILES_FOLDER) if f.endswith('.png') and f[:-4].isdigit()]
files.sort()

# GUI
class TileLabeler(tk.Tk):
    def __init__(self, files):
        super().__init__()
        self.title("Kingdomino Tile Labeler")
        self.geometry("500x300")
        self.files = files
        self.index = 0
        self.selected_tile = tk.StringVar(value=TILE_CLASSES[0])
        self.selected_crown = tk.IntVar(value=0)
        self.image_label = None
        self.name_label = None
        self.current_img = None
        self.current_path = None
        self._build_ui()
        self._load_image()

    def _build_ui(self):
        # Image display
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)
        # Tile type buttons
        frame_type = tk.Frame(self)
        frame_type.pack()
        for t in TILE_CLASSES:
            btn = ttk.Radiobutton(frame_type, text=CODE2TERR[t], variable=self.selected_tile, value=t, command=self._update_name)
            btn.pack(side=tk.LEFT, padx=2)
        # Crowns buttons
        frame_crown = tk.Frame(self)
        frame_crown.pack()
        for c in CROWN_CLASSES:
            btn = ttk.Radiobutton(frame_crown, text=str(c), variable=self.selected_crown, value=c, command=self._update_name)
            btn.pack(side=tk.LEFT, padx=2)
        # Name display
        self.name_label = tk.Label(self, text="", font=("Arial", 14))
        self.name_label.pack(pady=10)
        # Save button
        save_btn = ttk.Button(self, text="Save and Next", command=self._save_and_next)
        save_btn.pack(pady=10)

    def _load_image(self):
        if self.index >= len(self.files):
            self.name_label.config(text="DONE!")
            self.image_label.config(image='')
            return
        fname = self.files[self.index]
        path = os.path.join(TILES_FOLDER, fname)
        self.current_path = path
        img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        self.current_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.current_img)
        self._update_name()

    def _update_name(self):
        tile = self.selected_tile.get()
        crowns = self.selected_crown.get()
        next_name = get_next_free_name(tile, crowns)
        self.name_label.config(text=f"Will save as: {next_name}")

    def _save_and_next(self):
        tile = self.selected_tile.get()
        crowns = self.selected_crown.get()
        next_name = get_next_free_name(tile, crowns)
        # Save/copy file
        new_path = os.path.join(TILES_FOLDER, next_name)
        shutil.move(self.current_path, new_path)
        print(f"Saved as {next_name}")
        self.index += 1
        self._load_image()

if __name__ == "__main__":
    app = TileLabeler(files)
    app.mainloop()
