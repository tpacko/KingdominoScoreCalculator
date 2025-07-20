import os
import glob
import csv
import re
from tkinter import Tk, Canvas, Button, Label, Frame, Checkbutton, IntVar
from PIL import Image, ImageTk

ANNOTATION_FILE = "annotations.txt"
MAX_DISPLAY_SIZE = 800

CODE2TERR = {"f": "forest", "me": "meadow", "mi": "mine",
             "w": "water", "wa": "wasteland", "wh": "wheat", "c": "castle"}

class BoardAnnotator:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.index = 0
        self.annotations = self.load_annotations()
        self.points = []
        self.scale = 1.0
        self.rotated = False
        self.orig_size = None
        self.rotated_size = None
        self.skip_annotated = IntVar(value=0)

        # Canvas (size will be set dynamically)
        self.canvas = Canvas(root)
        self.canvas.pack(side="top")

        # Button panel
        btn_frame = Frame(root)
        btn_frame.pack(side="top")
        Button(btn_frame, text="Reset", command=self.reset).pack(side="left")
        Button(btn_frame, text="Skip", command=self.skip).pack(side="left")
        self.confirm_btn = Button(btn_frame, text="Confirm", command=self.confirm, state="disabled")
        self.confirm_btn.pack(side="left")
        Checkbutton(btn_frame, text="Skip all annotated", variable=self.skip_annotated).pack(side="left")

        # Status label
        self.status = Label(root, text="")
        self.status.pack(side="top")

        self.canvas.bind("<Button-1>", self.on_click)
        self.load_image()

    def load_annotations(self):
        if not os.path.exists(ANNOTATION_FILE):
            return {}
        with open(ANNOTATION_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile)
            return {row[0]: list(map(int, row[1:])) for row in reader}

    def save_annotations(self):
        with open(ANNOTATION_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for fname, pts in self.annotations.items():
                writer.writerow([fname] + pts)

    def find_next_index(self, start_idx=0):
        if not self.skip_annotated.get():
            return start_idx
        for idx in range(start_idx, len(self.image_paths)):
            fname = os.path.basename(self.image_paths[idx])
            if fname not in self.annotations:
                return idx
        return len(self.image_paths)

    def load_image(self):
        self.points = []
        self.index = self.find_next_index(self.index)
        if self.index >= len(self.image_paths):
            self.status.config(text="All images annotated.")
            self.canvas.delete("all")
            return

        self.filename = os.path.basename(self.image_paths[self.index])
        img = Image.open(self.image_paths[self.index])
        self.orig_size = img.size
        self.rotated = False

        if img.height > img.width:
            img = img.rotate(-90, expand=True)
            self.rotated = True

        self.rotated_size = img.size

        scale = MAX_DISPLAY_SIZE / img.width
        self.scale = scale
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)

        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.status.config(text=f"Annotating {self.filename}")

        if self.filename in self.annotations:
            pts = self.annotations[self.filename]
            orig_pts = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
            if self.rotated:
                orig_w, orig_h = self.orig_size
                rot_pts = [(orig_h - y0, x0) for (x0, y0) in orig_pts]
            else:
                rot_pts = orig_pts
            self.points = [self.scale_coords(pt) for pt in rot_pts]
            self.redraw()

    def redraw(self):
        self.canvas.delete("points")
        for i, pt in enumerate(self.points):
            x, y = pt
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="green", tags="points")
            if i > 0:
                self.canvas.create_line(self.points[i - 1], pt, fill="blue", tags="points")
        if len(self.points) == 4:
            self.canvas.create_line(self.points[3], self.points[0], fill="blue", tags="points")
            self.confirm_btn.config(state="normal")
        else:
            self.confirm_btn.config(state="disabled")

    def scale_coords(self, pt):
        return round(pt[0] * self.scale), round(pt[1] * self.scale)

    def descale_coords(self, pt):
        return pt[0] / self.scale, pt[1] / self.scale

    def on_click(self, event):
        if len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.redraw()

    def reset(self):
        self.points = []
        self.redraw()

    def skip(self):
        self.index += 1
        self.index = self.find_next_index(self.index)
        self.load_image()

    def confirm(self):
        if len(self.points) != 4:
            return
        pts_rot = [self.descale_coords(p) for p in self.points]
        if self.rotated:
            orig_w, orig_h = self.orig_size
            orig_pts = [(y_r, orig_h - x_r) for (x_r, y_r) in pts_rot]
        else:
            orig_pts = pts_rot
        flat = [int(round(c)) for pt in orig_pts for c in pt]
        self.annotations[self.filename] = flat
        self.save_annotations()
        self.index += 1
        self.index = self.find_next_index(self.index)
        self.load_image()

def main():
    input_dir = "files"
    pattern = re.compile(r'^game\d+\.')
    image_files = sorted([
        f for f in glob.glob(os.path.join(input_dir, '*.*'))
        if pattern.match(os.path.basename(f))
    ])

    if not image_files:
        print("No valid game images found in 'files/'")
        return

    root = Tk()
    root.title("Kingdomino Board Annotator")
    BoardAnnotator(root, image_files)
    root.mainloop()

if __name__ == "__main__":
    main()
