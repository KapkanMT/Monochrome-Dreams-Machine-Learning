#TOATE DATELE TREBUIESC BAGATE INTR-UN FOLDER DATA.

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)


#Functia de generare a celor 9 foldere corespunzatoare claselor, de asemenea returneaza un dataframe pentru fiecare
#tip de data: train, validation si test.


def setup(generic_path, label_path, is_test):
    col_names = ["id", "label"]
    labels_list = pd.read_csv(label_path, names=col_names)
    os.chdir(generic_path)
    if not is_test:
        if os.path.isdir('0') is False:
            for i in range(9):
                if os.path.isdir(f'{i}') is False:
                    os.makedirs(f'{i}')
            for i, row in labels_list.iterrows():
                photo = labels_list.iloc[i, 0]
                label = labels_list.iloc[i, 1]
                shutil.copy(photo, f'{label}')
    else:
        if os.path.isdir('unknown') is False:
            os.makedirs('unknown')
            for i, row in labels_list.iterrows():
                shutil.copy(labels_list.iloc[i, 0], 'unknown')
        del labels_list["label"]
    os.chdir('../../')
    return labels_list

#functie de plotare a imaginilor.
#luata de pe site-ul Tensorflow: https://www.tensorflow.org/tutorials/images/classification#visualize_training_images

def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#INCEPUTUL MAIN-ULUI----------------------------------------------

train_path = 'data/train'
validation_path = 'data/validation'
test_path = 'data/test'

#Creez dataframe-uri usor de prelucrat.

train_dataframe = setup(train_path, "data/train.txt", False)
validation_dataframe = setup(validation_path, "data/validation.txt", False)
test_dataframe = setup(test_path, "data/test.txt", True)

#Creez batch-uri si aplic filtrul folosit in vgg16

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(
    directory=train_path,
    classes=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
    target_size=(32, 32),
    batch_size=100,
    shuffle=True)
validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(
    directory=validation_path,
    classes=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
    target_size=(32, 32),
    batch_size=100,
    shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(
    directory=test_path,
    classes=['unknown'],
    target_size=(32, 32),
    batch_size=10,
    shuffle=False)

#Chem functia de afisati imaginile dupa preprocesare

images, labels = next(train_batches)
plot_images(images)
print(labels)

#Creez modelul, il antrenez si fac predictiile. Detalii in cod.

model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=9, activation='softmax')
])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=validation_batches,
          validation_steps=len(validation_batches),
          epochs=20,
          verbose=2)


predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=1)

#transform one-hot encoder intrun int
predictions2 = (np.argmax(predictions, axis=1))

#scriu toate rezultatele intr-un csv. Acesta se afla in test/unknown.

os.chdir("data/test/unknown")
with open('sample_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    i = 0
    for x in predictions2:
        writer.writerow([test_dataframe.iloc[i, 0], x])
        i = i+1

