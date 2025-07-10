import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image
import numpy as np
import os
import time
import requests
from PIL import Image
from io import BytesIO

tf.keras.backend.clear_session()

img_size = (128, 128)
batch_size = 32
            
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/nvidia02/finalitymygoodness",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/nvidia02/finalitymygoodness",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

num_classes = 6

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
# ])

# convolutional network is required for the network to compile


m = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    # data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
    tf.keras.layers.BatchNormalization()
])

m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = m.fit(train_ds, validation_data=val_ds, epochs=2)

test_loss, test_acc = m.evaluate(val_ds, verbose=2)

probability_model= tf.keras.Sequential([m, layers.Softmax()])

img_path = "/home/nvidia02/protisTest.JPEG"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0) / 255.0  # Normalize

predictions = m.predict(img_array)
class_names = ['Animalia', 'Plantae', 'Fungi', 'Protista', 'Archaebacteria', 'Eubacteria']

predicted_class = class_names[np.argmax(predictions[0])]
print(f"Predicted: {predicted_class}")
print(f"predictions shape: {predictions.shape}")
print(f"predicted: {predictions}")
