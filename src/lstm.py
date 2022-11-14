import tensorflow as tf
import json
CONFIG_PATH = "src/train_params.json"


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


class LSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model_name = "LSTM"
        self.model = tf.keras.Sequential( 
            [
                tf.keras.layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", input_shape=(config.get("time_steps", 10), 3)),
                tf.keras.layers.Dense(units = 1, activation = "linear"),
                #layers.Dropout(0.5),
                tf.keras.layers.RepeatVector(2),
                tf.keras.layers.LSTM(units = config.get("hidden_lstm", 10), activation = "tanh", return_sequences=True),
                #layers.Dropout(0.5),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = 1, activation="sigmoid"))
            ]
        )

    def call(self, inputs):
        return self.model(inputs)
