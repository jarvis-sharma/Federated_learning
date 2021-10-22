import argparse
from typing import Dict, Tuple, cast

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common import Weights

import modelFunctions
from sklearn.metrics import classification_report

class CovidCoughDetectionClient(fl.client.KerasClient):
    
    def __init__(
        self,
        model: tf.keras.Model,
        X_train,X_test,y_train,y_test
    ):
        self.model = model
        self.x_train, self.y_train = X_train,y_train
        self.x_test, self.y_test = X_test,y_test

    def get_weights(self) -> Weights:
        return cast(Weights, self.model.get_weights())

    def fit(
        self, weights: Weights, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[Weights, int, int]:

        # Use provided weights to update local model
        self.model.set_weights(weights)

        filename='log33.csv'
        # history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

        # Train the local model using local dataset
        # self.model.fit(
        #     self.x_train,
        #     self.y_train,
        #     batch_size=32,
        #     epochs=50,
        #     callbacks=[history_logger],
            
        # )
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),batch_size=32,
            epochs=50)
        # print(history)
        y_pred = self.model.predict(self.x_test, batch_size=25, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)

        print(classification_report(self.y_test, y_pred_bool))
        return self.model.get_weights(), len(self.x_train), len(self.x_train)

    def evaluate(
        self, weights: Weights, config: Dict[str, fl.common.Scalar]
    ) -> Tuple[int, float, float]:

        # Update local model and evaluate on local dataset
        self.model.set_weights(weights)
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, batch_size=len(self.x_test), verbose=1
        )
        print("Updated model accuracy:",float(accuracy))

        #Save weights of model
        # self.model.save_weights('model.h5')
        self.model.save('model3.h5')
        y_pred = self.model.predict(self.x_test, batch_size=25, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)

        print(classification_report(self.y_test, y_pred_bool))
        # Return number of evaluation examples and evaluation result (loss/accuracy)
        return len(self.x_test), float(loss), float(accuracy)
def load_datasetx(dataset_path):
    with open(dataset_path,"rb") as f:
        X = np.load(f)
    return X 

def main() -> None:

    DATASET_PATH1 = './X_train3.npy'
    DATASET_PATH2 = './y_train3.npy'
    DEFAULT_SERVER_ADDRESS = "[::]:8080"
    # VALID_DATA_PATH = './server.npy'
    VALID_DATA_PATH1 = './X_test3.npy'
    VALID_DATA_PATH2 = './y_test3.npy'
    X_test3=load_datasetx(VALID_DATA_PATH1)
    y_test3=load_datasetx(VALID_DATA_PATH2) 
    # Load model and data
    model = modelFunctions.load_model((40,30,1),X_test3,y_test3)
    X_train, X_test, y_train, y_test = modelFunctions.load_data(
        DATASET_PATH1=DATASET_PATH1,DATASET_PATH2=DATASET_PATH2,VALID_DATA_PATH1=VALID_DATA_PATH1,VALID_DATA_PATH2=VALID_DATA_PATH2
    )

    # Start client
    client = CovidCoughDetectionClient(model, X_train,X_test,y_train,y_test)
    fl.client.start_keras_client(DEFAULT_SERVER_ADDRESS, client)


if __name__ == "__main__":
    main()