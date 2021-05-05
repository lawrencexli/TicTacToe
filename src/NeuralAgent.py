import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from tensorflow.python.keras.models import model_from_json


class NeuralAgent:

    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            json_file = open('../model/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("../model/model.h5")
            print("Model loaded successfully")
        except FileNotFoundError:
            print("No model found! Creating new model automatically...")
            self.create_model()

    def create_model(self):
        self.model = Sequential()
        # Add a bunch of networks
        self.model.add(LSTM(32, input_shape=(3, 3), return_sequences=True))
        self.model.add(LSTM(16, return_sequences=True))
        self.model.add(LSTM(8, return_sequences=False))
        self.model.add(Dense(2, kernel_initializer='normal', activation='linear'))
        self.model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        self.model.summary()

    def train(self, inputs, outputs):
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # Actual do training
        self.model.fit(inputs, outputs, epochs=500, batch_size=5, validation_split=0.05, verbose=0)
        scores = self.model.evaluate(inputs, outputs, verbose=1, batch_size=5)
        print('Accuracy: {}'.format(scores[1]))

    def save_model(self):
        model_json = self.model.to_json()
        with open("../model/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("../model/model.h5")
        print("Saved model to disk in saved_model folder")

    def plot(self, inputs, outputs):
        predict = self.model.predict(inputs)
        print(predict)
        plt.plot(outputs, predict - outputs, 'C2')
        plt.ylim(ymax=3, ymin=-3)
        plt.show()


if __name__ == "__main__":
    X = [[[0, 3, 3], [1, 3, 3], [3, 3, 3]], [[0, 0, 3], [1, 3, 3], [3, 3, 1]], [[0, 0, 0], [1, 3, 1], [3, 3, 1]]]
    X = np.array(X)
    print(X.shape)

    y = [1, 2, 3]
    y = np.array(y)
    print(y.shape)

    myAgent = NeuralAgent()
    myAgent.load_model()
    myAgent.train(X, y)
    myAgent.plot(X, y)
