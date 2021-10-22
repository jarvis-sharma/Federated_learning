from typing import Tuple, cast
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import math
import random
import os
from keras import backend as K
from sklearn.metrics import classification_report
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
tf.get_logger().setLevel("ERROR")


def load_model(input_shape ,X_test, y_test) -> tf.keras.Model:

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
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    # output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

    #compile Model

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def load_dataset(dataset_path):
    with open(dataset_path,"rb") as f:
        X = np.load(f)
        y = np.load(f)
    return X,y

def load_datasetx(dataset_path):
    with open(dataset_path,"rb") as f:
        X = np.load(f)
    return X   

def splitFunction(X,y,test_size):
    test_samples = math.ceil(test_size*X.shape[0])
    
    X_test = X[0:test_samples]
    y_test = y[0:test_samples]
    X_train = X[test_samples:]
    y_train = y[test_samples:]
    
    return X_train, X_test, y_train, y_test

def load_data(DATASET_PATH1,DATASET_PATH2,VALID_DATA_PATH1,VALID_DATA_PATH2):
    #load data
    X = load_datasetx(DATASET_PATH1)
    y=load_datasetx(DATASET_PATH2)
    X_v= load_datasetx(VALID_DATA_PATH1)
    y_v= load_datasetx(VALID_DATA_PATH2)

    X_train, _,y_train, _ = splitFunction(X,y,0)
    _,X_test,_,y_test = splitFunction(X_v,y_v,1)
    #make into 3D array
    
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    return X_train, X_test, y_train, y_test


