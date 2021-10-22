import json
import numpy as np
from sklearn.model_selection import train_test_split
import random
import math
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET_PATH = "./client1.npy"



def load_dataset(dataset_path):
    with open(dataset_path,"rb") as f:
        X = np.load(f)
        y = np.load(f)
    return X,y

def splitFunction(X,y,test_size):
    test_samples = math.ceil(test_size*X.shape[0])
    
    X_test = X[0:test_samples]
    y_test = y[0:test_samples]
    X_train = X[test_samples:]
    y_train = y[test_samples:]
    
    return X_train, X_test, y_train, y_test

def load_data(test_size):
    #load data
    X,y = load_dataset(DATASET_PATH)
        
    X_train, X_test,y_train, y_test = splitFunction(X,y,test_size)

    #make into 3D array
    
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data(0.1)


def build_model(input_shape):
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model



input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
model = build_model(input_shape)



optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, batch_size=32, epochs=50)
loss, accuracy = model.evaluate(
        X_test, y_test, batch_size=len(X_test), verbose=1
    )
print("Accuracy is:",float(accuracy))

