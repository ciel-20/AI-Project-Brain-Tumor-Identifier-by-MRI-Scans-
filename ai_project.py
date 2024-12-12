import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Input

# Dataset Path (relative to the script)
base_dir = './dataset'

# List of labels (subfolder names)
labels = ['GliomaTumor', 'MeningiomaTumor', 'NoTumor', 'PituitaryTumor']

# Image size and arrays to hold the data
image_size = 150
X_train = []
Y_train = []

# Load training images from the dataset
for label in labels:
    folder_path = os.path.join(base_dir, 'Training', label)
    if os.path.exists(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                X_train.append(img)
                Y_train.append(label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    else:
        print(f"Folder {folder_path} does not exist.")

# Convert to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Shuffle the data
X_train, Y_train = shuffle(X_train, Y_train, random_state=101)

# Convert labels to categorical
y_train_new = [labels.index(label) for label in Y_train]
y_train = tf.keras.utils.to_categorical(y_train_new)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=101)

# Define the CNN model
model = Sequential([
    Input(shape=(150, 150, 3)),  # Define the input layer explicitly
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.3),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # Output layer for 4 classes
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Save the trained model
model.save('braintumor.h5')

# Plot accuracy and loss
import matplotlib.pyplot as plt

# Accuracy Plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Loss Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
