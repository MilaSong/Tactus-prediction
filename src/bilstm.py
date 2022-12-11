import tensorflow as tf
from tensorflow.keras import layers
import json
CONFIG_PATH = "src/train_params.json"


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


class BiLSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model_name = "BiLSTM"
        self.model = tf.keras.Sequential( 
            [
                layers.Bidirectional(layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", input_shape=(config.get("time_steps", 10), 3))),
                layers.Dense(units = 1, activation = "linear"),
                #layers.Dropout(0.5),
                layers.RepeatVector(2),
                layers.Bidirectional(layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", return_sequences=True)),
                #layers.Dropout(0.5),
                layers.TimeDistributed(layers.Dense(units = 1, activation="linear"))
            ]
        )

    def call(self, inputs):
        return self.model(inputs)

