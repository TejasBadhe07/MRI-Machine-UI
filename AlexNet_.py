import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import random

im_size = 227
DATADIR = r"D:\TEJAS\Projects\ECG\Training Images\Dataset_"
model_path = "ECGModel(7_Classes)"
CATEGORIES = ['Bigeminy', 'Isolated', 'Trigeminy', 'Noise', 'Normal', 'RonTtype', 'Vpair']
def create_alexnet():
    model = keras.Sequential()
    model.add(layers.Conv2D(96, 11, strides=4, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(im_size, im_size, 3)))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(384, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(CATEGORIES), activation='softmax'))
    return model

model = create_alexnet()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
a = model.summary()
print(a)

def preprocess_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (im_size, im_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, im_size, im_size, 3)
    y = np.array(y)

    return X, y


# Preprocess the data
X, y = preprocess_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# Convert the labels to one-hot encoded arrays
num_classes = len(CATEGORIES)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Evaluate the model on the test data
scores = model.evaluate(X_val, y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#model.save('model.h5')
tf.keras.models.save_model(model, model_path)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']