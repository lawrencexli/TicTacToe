import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import matplotlib.pyplot as plt

class NeuralAgent:

    def __init__(self):
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(8, input_shape=(5, 1), return_sequences=False))
        self.model.add(Dense(2, kernel_initializer='normal', activation ='linear'))
        self.model.add(Dense(1, kernel_initializer='normal', activation ='linear'))

    def train(self, inputs, outputs):
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.fit(inputs, outputs, epochs=500, batch_size=5, validation_split=0.05, verbose=0)
        scores = self.model.evaluate(inputs, outputs, verbose=1, batch_size=5)
        print('Accuracy: {}'.format(scores[1]))

    def plot(self, inputs, outputs):
        predict = self.model.predict(inputs)
        print(predict)
        plt.plot(outputs, predict - outputs, 'C2')
        plt.ylim(ymax=3, ymin=-3)
        plt.show()

if __name__ == "__main__":

    X = [[i + j for j in range(5)] for i in range(100)]
    X = np.array(X)

    y = [[i + (i - 1) * .5 + (i - 2) * .2 + (i - 3) * .1 for i in range(4, 104)]]
    y = np.array(y)
    X = X.reshape((100, 5, 1))
    y = y.reshape((100, 1))

    myAgent = NeuralAgent()
    myAgent.create_model()
    myAgent.train(X,y)
    myAgent.plot(X,y)




