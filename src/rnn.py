import tensorflow as tf
import json
CONFIG_PATH = "src/train_params.json"


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


class RNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model_name = "RNN"
        self.model = tf.keras.Sequential( 
            [
                tf.keras.layers.SimpleRNN(units = config.get("hidden_rnn", 10), activation = "tanh", input_shape=(config.get("time_steps", 10), 3)),
                #layers.Dropout(0.5),
                tf.keras.layers.Dense(units = 2, activation="linear")
            ]
        )

    def call(self, inputs):
        return self.model(inputs)
