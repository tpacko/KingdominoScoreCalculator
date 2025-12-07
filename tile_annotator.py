import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import shutil
import torch
from torchvision import transforms
from tile_learn import TILE_CLASSES, CROWN_CLASSES, CODE2TERR, IMG_SIZE
from model import TileNet

# Config
TILES_FOLDER = 'tiles'
ANNOTATIONS_FILE = os.path.join(TILES_FOLDER, 'annotations.txt')

# Utility for annotations management
def load_annotated_filenames():
    annotated = set()
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                # Expect: filename,type,crowns,x1;y1;x2;y2;x3;y3
                fname = parts[0].strip()
                annotated.add(fname)
    return annotated

def append_annotation(filename, tile_code, crowns, points):
    # points is list of up to 3 (x,y) in original image coords; pad with (-1,-1)
    padded = points[:3] + [(-1, -1)] * (3 - len(points))
    # We need x;y;x;y;x;y
    pts_order = []
    for (x, y) in padded:
        pts_order.append(str(x))
        pts_order.append(str(y))
    pts_field = ';'.join(pts_order)
    line = f"{filename},{tile_code},{crowns},{pts_field}\n"
    with open(ANNOTATIONS_FILE, 'a', encoding='utf-8') as f:
        f.write(line)

# Determine next free name staying within naming scheme
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

# Parse tile/crown from filename if already named correctly
def parse_tile_from_name(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    # formats: <tile><n> or <tile><n>_c<crowns>
    parts = base.split('_')
    head = parts[0]
    crowns = 0
    if len(parts) > 1 and parts[1].startswith('c'):
        try:
            crowns = int(parts[1][1:])
        except:
            crowns = 0
    # tile code is leading letters until digits
    i = 0
    while i < len(head) and not head[i].isdigit():
        i += 1
    tile_code = head[:i]
    if tile_code in TILE_CLASSES:
        return tile_code, crowns
    return None, None

# Build list of files needing labeling: those missing in annotations.txt
all_tile_files = [f for f in os.listdir(TILES_FOLDER) if f.endswith('.png')]
all_tile_files.sort()  # Ensure alphabetical order
annotated_set = load_annotated_filenames()
files = [f for f in all_tile_files if f not in annotated_set]
files.sort()  # Sort unannotated files alphabetically

# GUI
class TileLabeler(tk.Tk):
    def __init__(self, files):
        super().__init__()
        self.title("Kingdomino Tile Labeler")
        self.geometry("600x600")
        self.files = files
        self.index = 0
        self.selected_tile = tk.StringVar(value=TILE_CLASSES[0])
        self.selected_crown = tk.IntVar(value=0)
        self.image_label = None
        self.name_label = None
        self.current_img = None
        self.current_path = None
        self.suggestion_label = None
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Crown points (display coords) and original image size for mapping
        self.crown_points_display = []  # list of (x,y) in display canvas coords
        self.orig_size = None  # (w,h)
        self._load_classifier()
        self._build_ui()
        self._load_image()

    def _load_classifier(self):
        import os
        model_path = 'tile_classifier.pt'
        if os.path.exists(model_path):
            self.classifier = TileNet(len(TILE_CLASSES), len(CROWN_CLASSES)).to(self.device)
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            self.classifier.eval()
        else:
            self.classifier = None

    def _build_ui(self):
        # Image display using Canvas to draw crown centers
        self.canvas = tk.Canvas(self, width=IMG_SIZE, height=IMG_SIZE, bg="#222")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        # Tile type buttons
        frame_type = tk.Frame(self)
        frame_type.pack()
        for t in TILE_CLASSES:
            btn = ttk.Radiobutton(frame_type, text=CODE2TERR[t], variable=self.selected_tile, value=t, command=self._update_name)
            btn.pack(side="left", padx=2)
        # Crowns buttons
        frame_crown = tk.Frame(self)
        frame_crown.pack()
        for c in CROWN_CLASSES:
            btn = ttk.Radiobutton(frame_crown, text=str(c), variable=self.selected_crown, value=c, command=self._update_name)
            btn.pack(side="left", padx=2)
        # Reset crowns button
        reset_btn = tk.Button(self, text="Reset crowns", command=self._reset_crowns)
        reset_btn.pack(pady=5)
        # Name display
        self.name_label = tk.Label(self, text="", font=("Arial", 14))
        self.name_label.pack(pady=10)
        # Suggestion display
        self.suggestion_label = tk.Label(self, text="", font=("Arial", 12), fg="blue")
        self.suggestion_label.pack(pady=5)
        # Save button
        save_btn = tk.Button(self, text="Save and Next", command=self._save_and_next, font=("Arial", 16), height=2, width=18, bg="#4CAF50", fg="white", activebackground="#357a38")
        save_btn.pack(pady=20)

    def _predict_tile(self, img_path):
        if self.classifier is None:
            return None, None
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tile_logits, crown_logits = self.classifier(img_tensor)
            tile_idx = tile_logits.argmax(1).item()
            crown_idx = crown_logits.argmax(1).item()
        return tile_idx, crown_idx

    def _draw_image(self, path):
        # Load original to know size
        img = Image.open(path).convert('RGB')
        self.orig_size = img.size  # (w,h)
        disp = img.resize((IMG_SIZE, IMG_SIZE))
        self.current_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_img)
        self._redraw_crowns()

    def _redraw_crowns(self):
        # Clear previous crown drawings by deleting items with tag 'crown'
        self.canvas.delete('crown')
        r = 4
        for (x, y) in self.crown_points_display[:3]:
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline='yellow', fill='yellow', tags='crown')
        # Update crown count to number of points (cap 3)
        crowns = min(3, len(self.crown_points_display))
        self.selected_crown.set(crowns)
        self._update_name()

    def _on_canvas_click(self, event):
        # Allow only three points; ignore additional clicks
        if len(self.crown_points_display) >= 3:
            return
        x, y = event.x, event.y
        # Clamp within canvas
        x = max(0, min(IMG_SIZE - 1, x))
        y = max(0, min(IMG_SIZE - 1, y))
        self.crown_points_display.append((x, y))
        self._redraw_crowns()

    def _reset_crowns(self):
        self.crown_points_display = []
        self._redraw_crowns()

    def _load_image(self):
        if self.index >= len(self.files):
            self.name_label.config(text="DONE!")
            # Clear canvas image and crowns
            self.canvas.delete('all')
            self.suggestion_label.config(text="")
            return
        fname = self.files[self.index]
        path = os.path.join(TILES_FOLDER, fname)
        self.current_path = path
        # Draw image on canvas
        self._draw_image(path)
        # Suggestion from classifier
        tile_idx, crown_idx = self._predict_tile(path)
        if tile_idx is not None and crown_idx is not None:
            suggestion = f"Suggestion: {CODE2TERR[TILE_CLASSES[tile_idx]]}, {CROWN_CLASSES[crown_idx]}c"
            self.suggestion_label.config(text=suggestion)
            # Prefill type from filename if correctly named; otherwise from classifier
            tile_from_name, crowns_from_name = parse_tile_from_name(fname)
            if tile_from_name is not None:
                self.selected_tile.set(tile_from_name)
                # Crown count from name is only a starting point; clicks override
                self.selected_crown.set(crowns_from_name)
            else:
                self.selected_tile.set(TILE_CLASSES[tile_idx])
                self.selected_crown.set(CROWN_CLASSES[crown_idx])
        else:
            # No classifier; still try prefill from filename
            tile_from_name, crowns_from_name = parse_tile_from_name(fname)
            if tile_from_name is not None:
                self.selected_tile.set(tile_from_name)
                self.selected_crown.set(crowns_from_name)
            else:
                self.suggestion_label.config(text="No classifier suggestion available.")
        self._update_name()

    def _update_name(self):
        tile = self.selected_tile.get()
        crowns = self.selected_crown.get()
        next_name = get_next_free_name(tile, crowns)
        self.name_label.config(text=f"Will save as: {next_name}")

    def _display_to_original_coords(self, pts):
        # Map display (IMG_SIZE, IMG_SIZE) coordinates to original image pixels
        if not self.orig_size:
            return []
        ow, oh = self.orig_size
        scale_x = ow / IMG_SIZE
        scale_y = oh / IMG_SIZE
        mapped = []
        for (x, y) in pts[:3]:
            mapped.append((int(round(x * scale_x)), int(round(y * scale_y))))
        return mapped

    def _save_and_next(self):
        tile = self.selected_tile.get()
        # Crown count follows clicks (cap 3)
        crowns = min(3, len(self.crown_points_display))
        fname = os.path.basename(self.current_path)
        tile_from_name, crowns_from_name = parse_tile_from_name(fname)
        # If file is correctly named and crown count matches, keep name
        if tile_from_name == tile and crowns_from_name == crowns:
            save_name = fname
            new_path = self.current_path
        else:
            save_name = get_next_free_name(tile, crowns)
            new_path = os.path.join(TILES_FOLDER, save_name)
            shutil.move(self.current_path, new_path)
        # Prepare annotation line
        points_original = self._display_to_original_coords(self.crown_points_display)
        append_annotation(save_name, tile, crowns, points_original)
        print(f"Saved as {save_name} and annotated")
        # Move to next file
        self.index += 1
        self.crown_points_display = []
        self._load_image()

if __name__ == "__main__":
    # Ensure tiles folder exists and annotations file exists inside it
    os.makedirs(TILES_FOLDER, exist_ok=True)
    if not os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            f.write('')
    app = TileLabeler(files)
    app.mainloop()
