import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

# Load the model
model = load_model('digitclassification.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to smooth the digit image
def smooth_digit_image(cropped_image):
    blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_img = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.medianBlur(morphed, 3)
    resized = cv2.resize(smoothed, (28, 28), interpolation=cv2.INTER_AREA)
    return resized

# Dynamic Canny edge detection based on image variance
def dynamic_canny_threshold(image):
    variance = np.var(image)
    threshold1 = int(variance * 0.1)
    threshold2 = int(variance * 0.3)
    threshold1 = np.clip(threshold1, 50, 100)
    threshold2 = np.clip(threshold2, 150, 300)
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

# GUI Application
class DigitClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Classifier")
        
        # GUI Components
        self.label = Label(master, text="Digit Classifier", font=("Helvetica", 16))
        self.label.pack()

        self.canvas = Canvas(master, width=300, height=300, bg="white")
        self.canvas.pack()

        self.select_button = Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.result_label = Label(master, text="", font=("Helvetica", 14))
        self.result_label.pack()

        self.image_path = None
        self.processed_image = None

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if self.image_path:
            # Display selected image
            img = Image.open(self.image_path)
            img.thumbnail((300, 300))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=self.tk_image)

    def predict_digit(self):
        if not self.image_path:
            self.result_label.config(text="No image selected!")
            return

        # Read and preprocess the image
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]

        if height > 28 or width > 28:
            edges = dynamic_canny_threshold(image)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_height, image_width = edges.shape[:2]
            image_area = image_height * image_width
            min_area_ratio = 0.001
            max_area_ratio = 0.1
            min_area = min_area_ratio * image_area
            max_area = max_area_ratio * image_area

            valid_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

            largest_contour = max(valid_contours, key=cv2.contourArea) if valid_contours else None

            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                padding = 30
                x_start = max(x - padding, 0)
                y_start = max(y - padding, 0)
                x_end = min(x + w + padding, image.shape[1])
                y_end = min(y + h + padding, image.shape[0])
                cropped = image[y_start:y_end, x_start:x_end]
                self.processed_image = smooth_digit_image(cropped)
            else:
                self.processed_image = smooth_digit_image(image)
        else:
            _, self.processed_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Normalize and reshape for prediction
        image_normalized = self.processed_image / 255.0
        img = image_normalized.reshape(1, 28, 28)

        # Make predictions
        predictions = model.predict(img)
        predicted_class = predictions.argmax(axis=1)[0]
        self.result_label.config(text=f"Predicted Digit: {predicted_class}")

        # Display processed image
        plt.imshow(self.processed_image, cmap="gray")
        plt.show()

# Main Loop
if __name__ == "__main__":
    root = Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
