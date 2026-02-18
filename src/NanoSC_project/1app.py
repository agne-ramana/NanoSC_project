import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from NanoSC_project import TwoLayerCNN
        
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Draw")
        self.root.geometry("1400x500")

        self.logical_height = 28
        self.logical_width = 140
        self.scale_factor = 10
        self.canvas_width = self.logical_width * self.scale_factor
        self.canvas_height = self.logical_height * self.scale_factor

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

        # Image object to draw on (in memory)
        self.image = Image.new("L", (self.logical_width, self.logical_height), 255)

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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        self.model = TwoLayerCNN().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("../../models/model1.pt", map_location=self.device))
            self.model.eval()
            print("TwoLayerCNN Model loaded successfully.")
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
        if 0 <= gx < self.logical_width and 0 <= gy < self.logical_height:

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
        self.image = Image.new("L", (self.logical_width, self.logical_height), 255)
        self.result_label.config(text="Draw digits...", fg="black")


    def predict(self):
        # Convert PIL image to numpy array for OpenCV
        img_np = np.array(self.image)
        
        # Invert image
        img_inverted = 255 - img_np
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(img_inverted, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.result_label.config(text="Draw something first!", fg="red")
            return

        # Sort contours from left to right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Filter small noise (optional, e.g. single dots)
        contours_boxes = [(c, b) for c, b in zip(contours, bounding_boxes) if b[2] > 2 and b[3] > 2]
        
        if not contours_boxes:
            self.result_label.config(text="Draw clearly!", fg="red")
            return
            
        # Sort by x coordinate
        contours_boxes.sort(key=lambda x: x[1][0])
        
        results = []
        
        for c, bbox in contours_boxes:
            x, y, w, h = bbox
            # Crop the digit from the original PIL image
            digit_crop = self.image.crop((x, y, x + w, y + h))
            
            # Process the crop to be 28x28 centered
            img_processed = self.process_digit_segment(digit_crop)
            
            # Predict
            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(img_processed).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = output
                top_p, top_class = probs.topk(1, dim=1)
                certainty = top_p.item() * 100
                digit = top_class.item()
                results.append((digit, certainty))
        
        # Format output
        result_text = "Predictions:\n" + "\n".join([f"Digit: {d} ({c:.1f}%)" for d, c in results])
        self.result_label.config(text=result_text, fg="blue")

    def process_digit_segment(self, digit_img):
        # Invert to have white digit on black background
        digit_img = ImageOps.invert(digit_img)
        
        w, h = digit_img.size
        # Resize to fit in 20x20 box (preserve aspect ratio)
        max_dim = max(w, h)
        if max_dim == 0: return Image.new('L', (28, 28), 0)
        
        scale = 20.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        digit_resized = digit_img.resize((new_w, new_h), resample_method)
        
        # Create 28x28 black canvas
        final_img = Image.new('L', (28, 28), 0)
        
        # Center using Center of Mass
        digit_arr = np.array(digit_resized)
        total_mass = np.sum(digit_arr)
        
        if total_mass == 0:
            # Fallback to geometric center if mass is 0
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
        else:
            h_r, w_r = digit_arr.shape
            y_indices, x_indices = np.indices((h_r, w_r))
            cy = np.sum(y_indices * digit_arr) / total_mass
            cx = np.sum(x_indices * digit_arr) / total_mass
            
            target_center = 13.5
            paste_x = int(round(target_center - cx))
            paste_y = int(round(target_center - cy))
            
        final_img.paste(digit_resized, (paste_x, paste_y))
        return final_img


    def standardize_image(self, img):
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