import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog, Button, Label, messagebox

# -----------------------------
# Globals
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 10
EPOCHS = 3               # increased so model has a chance to learn (adjust as needed)
MODEL = None
CLASS_INDEX_TO_NAME = {}
class_names = []
TRAIN_DIR = None
TEST_DIR = None

# -----------------------------
# Utilities
# -----------------------------
def save_labels_file(save_path, class_order):
    label_file = os.path.splitext(save_path)[0] + "_labels.txt"
    with open(label_file, "w") as f:
        f.write("\n".join(class_order))

def check_class_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))

def deprocess_resnet(x):
    # x is a batch (N,H,W,3) preprocessed by tf.keras.applications.resnet50.preprocess_input (caffe mode)
    # reverse the preprocess for plotting: convert BGR back to RGB and add means
    x = x.copy()
    # if shape is (H,W,3) expand
    single = False
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
        single = True
    # Preprocess used: x[..., ::-1] - mean_BGR
    # So to reverse: add mean_BGR then convert BGR->RGB
    mean = np.array([103.939, 116.779, 123.68])  # B, G, R
    # add the mean
    x = x + mean.reshape((1,1,1,3))
    # BGR -> RGB
    x = x[..., ::-1]
    x = np.clip(x, 0, 255).astype(np.uint8)
    if single:
        return x[0]
    return x

# -----------------------------
# Model builder (ResNet50 transfer)
# -----------------------------
def build_resnet_transfer(num_classes, dropout=0.4, fine_tune=True):
    # Use imagenet weights
    base = ResNet50(weights='imagenet', include_top=False,
                    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    if not fine_tune:
        base.trainable = False
    else:
        # unfreeze last N layers for fine-tuning; here we unfreeze last 30 layers to give more capacity
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers[-30:]:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    # lower lr because fine-tuning
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Training
# -----------------------------
def train_model_gui():
    global MODEL, CLASS_INDEX_TO_NAME, class_names, TRAIN_DIR, EPOCHS

    TRAIN_DIR = filedialog.askdirectory(title="Select Training Folder (with subfolders)", initialdir=os.getcwd())
    if not TRAIN_DIR:
        return

    # Detect classes
    class_names[:] = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    if not class_names:
        messagebox.showerror("Error", "No class subfolders found in selected folder.")
        return

    messagebox.showinfo("Info", f"Detected classes: {', '.join(class_names)}")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Class mapping
    # Note: train_gen.class_indices maps name->index; we want index->name
    name_to_index = train_gen.class_indices
    CLASS_INDEX_TO_NAME = {idx: name for name, idx in name_to_index.items()}
    class_order = [CLASS_INDEX_TO_NAME[i] for i in range(len(CLASS_INDEX_TO_NAME))]
    print("Class index->name mapping:", CLASS_INDEX_TO_NAME)

    # Class balance
    counts = check_class_balance(train_gen.classes)
    print("Training class counts:", counts)
    y_for_cw = train_gen.classes
    # compute_class_weight requires labels as integers
    class_weights_array = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_for_cw),
                                         y=y_for_cw)
    class_weights = {i: float(class_weights_array[i]) for i in range(len(class_weights_array))}
    print("Computed class weights:", class_weights)

    # Build model - enable fine tuning to help learn classes other than 'patches' & 'pitting'
    MODEL = build_resnet_transfer(num_classes=len(class_names), dropout=0.5, fine_tune=True)

    # Callbacks - include checkpoint to retain best weights
    callbacks = [
        ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    steps_per_epoch = max(1, math.ceil(train_gen.samples / BATCH_SIZE))
    validation_steps = max(1, math.ceil(val_gen.samples / BATCH_SIZE))

    messagebox.showinfo("Training", f"Training ResNet50-based model for up to {EPOCHS} epochs.\nThis may take time.")
    history = MODEL.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Plot history
    plot_history(history)

    # Evaluate
    evaluate_and_report(MODEL, val_gen)

    # Save model
    save_path = filedialog.asksaveasfilename(defaultextension=".h5",
                                             filetypes=[("Keras Model", "*.h5")],
                                             title="Save trained model as",
                                             initialdir=os.getcwd())
    if save_path:
        MODEL.save(save_path)
        save_labels_file(save_path, class_order)
        messagebox.showinfo("Saved", f"Model and label file saved:\n{save_path}")
    else:
        messagebox.showwarning("Warning", "Model not saved.")

# -----------------------------
# Load model
# -----------------------------
def load_model_gui():
    global MODEL, CLASS_INDEX_TO_NAME, class_names
    model_path = filedialog.askopenfilename(title="Select saved .h5 model", filetypes=[("Keras Model", "*.h5")])
    if not model_path:
        return
    try:
        MODEL = load_model(model_path)
        label_file = os.path.splitext(model_path)[0] + "_labels.txt"
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                class_names[:] = [line.strip() for line in f.readlines()]
            CLASS_INDEX_TO_NAME = {i: name for i, name in enumerate(class_names)}
            print("Loaded labels:", CLASS_INDEX_TO_NAME)
        else:
            messagebox.showwarning("Warning", "Label file not found. Load test folder to set class names.")
        messagebox.showinfo("Loaded", f"Model loaded: {os.path.basename(model_path)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        MODEL = None

# -----------------------------
# Load test folder
# -----------------------------
def load_test_dir():
    global TEST_DIR, CLASS_INDEX_TO_NAME, class_names
    TEST_DIR = filedialog.askdirectory(title="Select Test Folder (with class subfolders)", initialdir=os.getcwd())
    if not TEST_DIR:
        return
    class_names[:] = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    CLASS_INDEX_TO_NAME = {i: name for i, name in enumerate(class_names)}
    messagebox.showinfo("Loaded", f"Test classes: {', '.join(class_names)}")

# -----------------------------
# Prediction
# -----------------------------
def predict_single_image_gui():
    img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")])
    if img_path:
        predict_single_image(img_path)

def predict_single_image(img_path, show_plot=True):
    if MODEL is None:
        messagebox.showerror("Error", "No model loaded. Load or train a model first.")
        return None

    try:
        img = keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        orig_arr = keras.preprocessing.image.img_to_array(img).astype(np.float32)  # original range 0-255
        arr = np.expand_dims(orig_arr.copy(), axis=0)
        arr_pre = resnet_preprocess(arr.copy())  # preprocessed for ResNet
        preds = MODEL.predict(arr_pre, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_INDEX_TO_NAME.get(idx, "Unknown")
        conf = float(preds[idx]) * 100.0

        if show_plot:
            disp = deprocess_resnet(arr_pre[0])  # reverse preprocessing for display
            plt.imshow(disp)
            plt.title(f"{label} ({conf:.2f}%)")
            plt.axis("off")
            plt.show()

        messagebox.showinfo("Prediction", f"{os.path.basename(img_path)}\nPredicted: {label}\nConfidence: {conf:.2f}%")
        return {"path": img_path, "label": label, "confidence": conf, "probs": preds}
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")
        return None

def predict_multiple_images_gui():
    files = filedialog.askopenfilenames(title="Select images", filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")])
    if not files:
        return
    results = []
    for f in files:
        res = predict_single_image(f, show_plot=False)
        if res:
            results.append(res)
    summary = "\n".join([f"{os.path.basename(r['path'])} -> {r['label']} ({r['confidence']:.2f}%)" for r in results])
    messagebox.showinfo("Batch predictions", summary)

# -----------------------------
# Evaluation & plotting
# -----------------------------
def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, generator):
    # Ensure we predict all samples by specifying steps
    steps = max(1, math.ceil(generator.samples / generator.batch_size))
    preds = model.predict(generator, steps=steps, verbose=1)
    y_true = generator.classes
    y_pred = np.argmax(preds, axis=1)

    # generator.class_indices maps class_name->index; we need index->name in order
    idx_to_name = {v:k for k,v in generator.class_indices.items()}
    # Build list of target names in index order
    target_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    print("Classification report:\n", classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(target_names)), max(6, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.show()

# -----------------------------
# GUI
# -----------------------------
def build_gui():
    root = Tk()
    root.title("Steel Surface Defect Detection (ResNet50 TL)")
    root.geometry("520x460")

    Label(root, text="Steel Surface Defect Detection", font=("Arial", 16, "bold")).pack(pady=8)
    Label(root, text="ResNet50 TL + Augmentation + Class Weights").pack(pady=4)

    Button(root, text="Train Model (select training folder)", width=40, command=train_model_gui).pack(pady=6)
    Button(root, text="Load Saved Model (.h5)", width=40, command=load_model_gui).pack(pady=6)
    Button(root, text="Load Test Folder (to set classes)", width=40, command=load_test_dir).pack(pady=6)
    Button(root, text="Predict Single Image", width=40, command=predict_single_image_gui).pack(pady=6)
    Button(root, text="Predict Multiple Images (summary)", width=40, command=predict_multiple_images_gui).pack(pady=6)
    Button(root, text="Exit", width=40, command=root.quit).pack(pady=18)

    root.mainloop()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
    build_gui()
