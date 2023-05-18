import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import numpy as np

# Load the EMNIST dataset
emnist_train, info = tfds.load(name='emnist', split='train', with_info=True)
emnist_test = tfds.load(name='emnist', split='test')

# Define the CNN architecture
model = keras.Sequential([
    layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(33, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 10
model.fit(emnist_train, epochs=epochs)

# Save the trained model
model.save('vin_model.h5')

# Load the trained model
model = keras.models.load_model('vin_model.h5')

# Define a function to preprocess the images
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='grayscale', target_size=(28, 28)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Define the ASCII character codes for each class
char_codes = {
    0: 65, 1: 66, 2: 67, 3: 68, 4: 69, 5: 70, 6: 71, 7: 72, 8: 74, 9: 75,
    10: 76, 11: 77, 12: 78, 13: 80, 14: 82, 15: 83, 16: 84, 17: 85, 18: 86,
    19: 87, 20: 88, 21: 89, 22: 90, 23: 48, 24: 49, 25: 50, 26: 51, 27: 52,
    28: 53, 29: 54, 30: 55, 31: 56, 32: 57
}

# Get the directory path from command-line arguments
image_directory = sys.argv[1]

# Find all image files in the directory
image_files = [
    os.path.join(image_directory, filename)
    for filename in os.listdir(image_directory)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Process each image and print the output
for image_file in image_files:
    image = preprocess_image(image_file)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    char_code = char_codes[predicted_class]
    print(f'{char_code}, {image_file}')