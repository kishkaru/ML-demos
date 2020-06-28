import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

DATADIR = "/home/kishan/Desktop/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100


def create_and_store_training_data(data_save_dir_name):
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        # get the classification  (0 or a 1). 0=dog 1=cat
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                # Convert image to array and grayscale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # Resize image to IMG_SIZExIMG_SIZE
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # Save data as [features, label] : [image, hot_hot vector]
                training_data.append([img_array, class_num])
                # plt.imshow(img_array, cmap='gray')  # graph it
                # plt.show()  # display!
            except Exception as e:
                pass
            # break
        # break

    print('Training dataset size:', len(training_data))
    # Shuffle training data so cat and dog training data is interleaved
    random.shuffle(training_data)

    features = []
    labels = []
    for f, l in training_data:
        features.append(f)
        labels.append(l)
    features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # last param is '1' for grayscale
    labels = np.array(labels)

    # Save dataset for later
    np.save(f'{data_save_dir_name}/features.npy', features)
    np.save(f'{data_save_dir_name}/labels.npy', labels)

    return data_save_dir_name


def train_model(training_data_path):
    # Load dataset
    features = np.load(f'{training_data_path}/features.npy')
    labels = np.load(f'{training_data_path}/labels.npy')

    # Normalize features data to values between [0, 1]
    features = features / 255.0

    dense_layers = [0, 1, 2]
    layer_sizes = [32, 64]
    conv_layers = [1, 2, 3]
    now = int(time.time())

    for dense_layer in dense_layers:
        for conv_layer in conv_layers:
            for layer_size in layer_sizes:
                NAME = f"Cats-vs-dogs-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{now}"
                print(NAME)
                tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

                # A feed forward model (go in forward order)
                model = Sequential()

                # layer 1: Convolutional (2D layer)
                # (nodes, window size (3x3), shape of data)
                model.add(Conv2D(64, (3, 3), input_shape=features.shape[1:]))
                model.add(Activation('relu'))
                # (2x2 pool size)
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # (Optional) Convolutional layer(s) (2D layers)
                for _ in range(conv_layer - 1):
                    model.add(Conv2D(64, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())  # convert 3D feature maps to 1D feature vectors

                # (Optional) Dense layer(s) (1D "fully connected" layer)
                for _ in range(dense_layer):
                    model.add(Dense(64))
                    model.add(Activation('relu'))

                # output Dense layer (1D "fully connected" layer)
                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                # Train model
                model.fit(features, labels, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])
                model_path = f'models/{NAME}.model'
                model.save(model_path)

    return model_path


def predict_using_model(model_name):
    def prepare(path):
        # Convert image to array and grayscale
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Resize image to IMG_SIZExIMG_SIZE
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # last param is '1' for grayscale

    for i in range(10):
        model = load_model(model_name)
        prediction = model.predict([prepare(f'{DATADIR}/Testing/Cat/{i}.jpg')])
        print(CATEGORIES[int(prediction[0][0])])


# 1) Create and store the training data as numpy arrays (as needed)
# training_data_path = create_and_store_training_data('cat_dog')

# 2) Train the model using loaded training data
# model_name = train_model(training_data_path)

# 3) Predict images using trained model
# predict_using_model(model_name)
predict_using_model("models/Cats-vs-dogs-3-conv-64-nodes-1-dense-1590134486.model")
