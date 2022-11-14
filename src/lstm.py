import tensorflow as tf
import json


config = {
    "hidden_lstm": 40,
    "time_steps": 40
}


class LSTMModel(tf.keras.Model):
    def __init__(self):

        self.model = tf.keras.Sequential( 
            [
                tf.layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", input_shape=(config.get("time_steps", 10), 3)),
                tf.layers.Dense(units = 1, activation = "linear"),
                #layers.Dropout(0.5),
                tf.layers.RepeatVector(2),
                tf.layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", return_sequences=True),
                #layers.Dropout(0.5),
                tf.layers.TimeDistributed(tf.layers.Dense(units = 1, activation="sigmoid"))
            ]
        )

    def call(self, inputs):
        return self.model(inputs)