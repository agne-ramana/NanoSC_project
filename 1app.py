import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()

        # Activation, normalisation & dropout functions
        self.relu = nn.ReLU() 
        self.bnorm32 = nn.BatchNorm2d(num_features=32)
        self.bnorm64 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.4)

        # Layer 1: Conv (1 input -> 32 output)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')

        # Layer 2: Conv (32 -> 64)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        
        # Layer 3: Fully Connected
        self.fc1 = nn.Linear(12544, 64)

        # Layer 4: Output
        self.out = nn.Linear(64, 10) 
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        # Layer 1
        x = self.relu(self.conv1(x))
        
        # Layer 2
        x = self.pool(self.relu(self.conv2(x)))
        
        # Dropout
        x = self.drop(x)
        
        # Flatten: (Batch, 64, 14, 14) -> (Batch, 12544)
        x = x.view(x.size(0), -1) 
        
        # Dense Layers
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.sigmoid(self.out(x))
        return x
        
# --- 2. The Application ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Draw")
        self.root.geometry("400x500")

        self.logical_size = 28
        self.scale_factor = 10
        self.canvas_width = self.logical_size * self.scale_factor
        self.canvas_height = self.logical_size * self.scale_factor

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.pack(pady=20)
        
        # Bindings for smooth drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.last_x = None
        self.last_y = None
        
        # Keep track of set pixels to avoid drawing overlap on canvas
        self.drawn_pixels = set()

        # Image object to draw on (in memory) - keep it at 28x28
        self.image = Image.new("L", (self.logical_size, self.logical_size), 255)
        self.draw_handle = ImageDraw.Draw(self.image)

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.predict_btn = tk.Button(self.btn_frame, text="Predict", command=self.predict, font=("Arial", 14), bg="#4CAF50", fg="white")
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(self.btn_frame, text="Clear", command=self.clear_canvas, font=("Arial", 14), bg="#f44336", fg="white")
        self.clear_btn.pack(side=tk.LEFT, padx=10)

        # Label for result
        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 16))
        self.result_label.pack(pady=20)
        
        # Load Model
        self.load_model()

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BinaryCNN().to(self.device)
        
        try:
            # Load the specific weights
            self.model.load_state_dict(torch.load("models/model1.pt", map_location=self.device))
            self.model.eval()
            print("BinaryCNN Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.result_label.config(text="Error loading model!", fg="red")

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw(event)

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.interpolate_and_draw(self.last_x, self.last_y, x, y)
        self.last_x = x
        self.last_y = y

    def interpolate_and_draw(self, start_x, start_y, end_x, end_y):
        dist = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        steps = int(dist) + 1
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            curr_x = start_x + (end_x - start_x) * t
            curr_y = start_y + (end_y - start_y) * t
            grid_x = int(curr_x // self.scale_factor)
            grid_y = int(curr_y // self.scale_factor)
            self.paint_pixel(grid_x, grid_y)
            self.paint_pixel(grid_x + 1, grid_y)
            self.paint_pixel(grid_x, grid_y + 1)
            self.paint_pixel(grid_x + 1, grid_y + 1)

    def paint_pixel(self, gx, gy):
        if 0 <= gx < self.logical_size and 0 <= gy < self.logical_size:
            if (gx, gy) not in self.drawn_pixels:
                self.drawn_pixels.add((gx, gy))
                x1 = gx * self.scale_factor
                y1 = gy * self.scale_factor
                x2 = x1 + self.scale_factor
                y2 = y1 + self.scale_factor
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")
                self.image.putpixel((gx, gy), 0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawn_pixels.clear()
        self.image = Image.new("L", (self.logical_size, self.logical_size), 255)
        self.result_label.config(text="Draw a digit...", fg="black")

    def predict(self):
        img_processed = self.standardize_image(self.image)
        
        transform = transforms.Compose([
            transforms.ToTensor(), # Converts 0-255 -> 0.0-1.0
        ])
        
        img_tensor = transform(img_processed).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)

            probs = output 
            
            top_p, top_class = probs.topk(1, dim=1)
            
            # This calculates the percentage (e.g., 0.95 -> 95.0%)
            certainty = top_p.item() * 100 
            digit = top_class.item()
            
        self.result_label.config(text=f"Prediction: {digit}\nCertainty: {certainty:.2f}%", fg="blue")

    def standardize_image(self, img):
        # Keeps your original high-quality centering logic
        img_inverted = ImageOps.invert(img)
        bbox = img_inverted.getbbox()
        if not bbox: return img_inverted 
        digit = img_inverted.crop(bbox)
        w, h = digit.size
        if w == 0 or h == 0: return img_inverted
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        digit_resized = digit.resize((new_w, new_h), resample_method)
        final_img = Image.new('L', (28, 28), 0)
        digit_arr = np.array(digit_resized)
        total_mass = np.sum(digit_arr)
        if total_mass == 0: return final_img
        h_r, w_r = digit_arr.shape
        y_indices, x_indices = np.indices((h_r, w_r))
        cy = np.sum(y_indices * digit_arr) / total_mass
        cx = np.sum(x_indices * digit_arr) / total_mass
        target_center = 13.5
        paste_x = int(round(target_center - cx))
        paste_y = int(round(target_center - cy))
        final_img.paste(digit_resized, (paste_x, paste_y))
        return final_img

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()