import json
import numpy as np
import random
import math
import tensorflow.keras as keras
import tensorflow as tf
import os
import flwr as fl

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_data(dataset_path):
    with open(dataset_path,"rb") as f:
        X = np.load(f)
        y = np.load(f)
    return X,y

def splitFunction(X,y,test_size,valid_size):
    test_samples = math.ceil(test_size*X.shape[0])
    valid_samples = math.ceil(valid_size*X.shape[0])
    
    X_test = X[0:test_samples]
    y_test = y[0:test_samples]
    X_valid = X[test_samples:valid_samples+test_samples]
    y_valid = y[test_samples:valid_samples+test_samples]
    X_train = X[valid_samples+test_samples:]
    y_train = y[valid_samples+test_samples:]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def prepare_datasets(test_size,valid_size):
    #load data
    X,y = load_data(DATASET_PATH)
    
    #create train test split
    X_new = []
    y_new = []
    
    j=0
    lenn=X.shape[0]
    while(j<=lenn-1):
        i = random.randint(0, X.shape[0]-1)
        X_new.append(X[i])
        y_new.append(y[i])
        
        X = np.delete(X, i, axis=0)
        y = np.delete(y, i, axis=0)

        j+=1
    
    X = np.array(X_new)
    y = np.array(y_new)
        
    #create train validation split
    
    X_train,X_valid,X_test,y_train,y_valid,y_test = splitFunction(X,y,test_size,valid_size)

    #make into 3D array
    
    X_train = X_train[...,np.newaxis]
    X_valid = X_valid[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

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


DATASET_PATH = "/home/jarvis/Downloads/client2.npy"

X,y = load_data(DATASET_PATH)
print(X.shape[0])

X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_datasets(0.2,0.25)

input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
model = build_model(input_shape)

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=100,verbose=0)
# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32,verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test,verbose=0)
        print("###############################2222222222222222222222222222")
        print(accuracy)
        print("###############################2222222222222222222222222222")
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient())        
