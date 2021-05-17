from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.models import model_from_json

"""
NeuralAgent class is a LSTM neural network model for predicting human behavior
"""

class NeuralAgent:

    def __init__(self):
        self.model = None

    """
    Try load model from disk. If the file does not exist, then create a new one.
    """
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

    """
    Create a new neural model
    """
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(9, input_shape=(3, 3), return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
        self.model.summary()

    """
    Train the network
    """
    def train(self, inputs, outputs):
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        # Actual do training using fit() method
        self.model.fit(inputs, outputs, epochs=500, batch_size=5, validation_split=0.05, verbose=0)
        scores = self.model.evaluate(inputs, outputs, verbose=1, batch_size=5)
        print('Accuracy: {}'.format(scores[1]))

    """
    Save the model
    """
    def save_model(self):
        model_json = self.model.to_json()
        with open("../model/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("../model/model.h5")
        print("Saved model to disk in model folder")

    """
    Predict actions
    """
    def predict(self, inputs):
        return self.model.predict(inputs)
