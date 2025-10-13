import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, messagebox

IMG_SIZE = 224
class_names = []
MODEL = None
TEST_DIR = None
CLASS_INDICES = {}  # maps model output index → class name

# =========================
# Load model dynamically
# =========================
def load_model_gui():
    global MODEL, CLASS_INDICES, class_names
    model_path = filedialog.askopenfilename(
        title="Select Steel Defect Model (.h5)",
        initialdir=os.getcwd(),
        filetypes=[("Keras Model", ".h5"), ("All files", ".*")]
    )
    if not model_path:
        messagebox.showerror("Error", "No model selected!")
        return
    try:
        MODEL = tf.keras.models.load_model(model_path)
        messagebox.showinfo("Success", f"Model loaded: {os.path.basename(model_path)}")
        
        # If test folder is already loaded, rebuild CLASS_INDICES
        if TEST_DIR:
            class_names.clear()
            class_names.extend(sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]))
            CLASS_INDICES = {i: name for i, name in enumerate(class_names)}
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        MODEL = None

# =========================
# Train model dynamically
# =========================
def train_model_gui():
    global MODEL, class_names, CLASS_INDICES
    train_dir = filedialog.askdirectory(title="Select Training Folder (contains class subfolders)",
                                        initialdir=os.getcwd())
    if not train_dir:
        messagebox.showerror("Error", "No training folder selected!")
        return

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not class_names:
        messagebox.showerror("Error", "No class subfolders found in training folder!")
        return

    messagebox.showinfo("Info", f"Classes detected: {', '.join(class_names)}")

    # Image data generators
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Build CLASS_INDICES mapping (index → class)
    CLASS_INDICES = {v: k for k, v in train_gen.class_indices.items()}

    # Simple CNN model
    MODEL = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])
    MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    messagebox.showinfo("Info", "Training started. This may take a while...")
    MODEL.fit(train_gen, validation_data=val_gen, epochs=10)

    # Save model
    save_path = filedialog.asksaveasfilename(defaultextension=".h5",
                                             filetypes=[("Keras Model", ".h5"), ("All files", ".*")],
                                             title="Save Trained Model As",
                                             initialdir=os.getcwd())
    if save_path:
        MODEL.save(save_path)
        messagebox.showinfo("Success", f"Model trained and saved: {save_path}")
    else:
        messagebox.showwarning("Warning", "Model not saved.")

# =========================
# Load test images folder
# =========================
def load_test_dir():
    global TEST_DIR, class_names, CLASS_INDICES
    TEST_DIR = filedialog.askdirectory(title="Select Test Images Folder", initialdir=os.getcwd())
    if not TEST_DIR:
        messagebox.showerror("Error", "No folder selected!")
        return

    class_names = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    CLASS_INDICES = {i: name for i, name in enumerate(class_names)}

    messagebox.showinfo("Success", f"Loaded classes: {', '.join(class_names)}")

# =========================
# Predict single image
# =========================
def predict_single_image(img_path):
    if not os.path.exists(img_path):
        messagebox.showerror("Error", f"Image not found:\n{img_path}")
        return

    img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    if MODEL is not None:
        try:
            predictions = MODEL.predict(processed_img, verbose=0)
            predicted_index = np.argmax(predictions)
            predicted_class = CLASS_INDICES.get(predicted_index, "Unknown")  # ✅ Safe mapping
            confidence = np.max(predictions)
            title_text = f"{predicted_class} ({confidence*100:.1f}%)"
        except Exception as e:
            title_text = f"Prediction Error: {e}"
    else:
        title_text = "Model not loaded – displaying image only"

    plt.imshow(img)
    plt.title(title_text)
    plt.axis('off')
    plt.show()

# =========================
# GUI functions
# =========================
def predict_single_image_gui():
    img_path = filedialog.askopenfilename(
        title="Select Image for Prediction",
        initialdir=os.getcwd(),
        filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp *.tif"), ("All files", ".*")]
    )
    if img_path:
        predict_single_image(img_path)

def predict_multiple_images_gui():
    file_paths = filedialog.askopenfilenames(
        title="Select Images for Prediction",
        initialdir=os.getcwd(),
        filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp *.tif"), ("All files", ".*")]
    )
    if not file_paths:
        return
    for img_path in file_paths:
        predict_single_image(img_path)

# =========================
# GUI Layout
# =========================
root = Tk()
root.title("Steel Defect Detection")
root.geometry("450x450")

Label(root, text="Steel Surface Defect Detection", font=("Arial", 14, "bold")).pack(pady=10)
Button(root, text="Load Model (.h5)", width=30, command=load_model_gui).pack(pady=5)
Button(root, text="Train Model (Option 2)", width=30, command=train_model_gui).pack(pady=5)
Button(root, text="Load Test Folder", width=30, command=load_test_dir).pack(pady=5)
Button(root, text="Predict Single Image", width=30, command=predict_single_image_gui).pack(pady=5)
Button(root, text="Predict Multiple Images", width=30, command=predict_multiple_images_gui).pack(pady=5)
Button(root, text="Exit", width=30, command=root.quit).pack(pady=20)

root.mainloop()