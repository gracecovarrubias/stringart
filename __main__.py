import cProfile
import pstats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from stringart import StringArtGenerator


class StringArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("String Art Generator")
        self.generator = StringArtGenerator()

        # Load Image Button
        self.load_button = tk.Button(
            root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Generate String Art Button
        self.generate_button = tk.Button(
            root, text="Generate String Art", command=self.generate_string_art)
        self.generate_button.pack()

        # Canvas for Displaying Image
        self.image_canvas = tk.Canvas(root, width=300, height=300)
        self.image_canvas.pack()

        # Matplotlib Figure for Displaying String Art
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        self.image_path = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
                                               )
        if file_path:
            self.image_path = file_path
            self.generator.load_image(file_path)
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            self.display_image(img)

    def display_image(self, img):
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(150, 150, image=self.img_tk)

    def generate_string_art(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        # Preprocess and generate the pattern
        self.generator.preprocess()
        self.generator.set_nails(180)
        self.generator.set_seed(42)
        self.generator.set_iterations(4000)
        pattern = self.generator.generate()

        # Extract lines from pattern
        lines_x = []
        lines_y = []
        for i, j in zip(pattern, pattern[1:]):
            lines_x.append((i[0], j[0]))
            lines_y.append((i[1], j[1]))

        # Plot on the Matplotlib canvas
        self.ax.clear()
        self.ax.axis('off')
        xmin, ymin = 0, 0
        xmax = self.generator.data.shape[0]
        ymax = self.generator.data.shape[1]
        self.ax.set_xlim([xmin, xmax])
        self.ax.set_ylim([ymin, ymax])
        self.ax.set_aspect('equal')

        batchsize = 10
        for i in range(0, len(lines_x), batchsize):
            self.ax.plot(lines_x[i:i + batchsize],
                         lines_y[i:i + batchsize], linewidth=0.1, color='k')
        self.canvas.draw()


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = StringArtApp(root)

    # Handle closing the window and exiting the program completely
    def on_closing():
        root.destroy()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
