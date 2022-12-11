import tensorflow as tf
from tensorflow.keras import layers
import json
from attention_class import attention
CONFIG_PATH = "src/train_params.json"


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


class RNNAttModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model_name = "RNN_attention"
        self.model = tf.keras.Sequential( 
            [
                layers.SimpleRNN(units = config.get("hidden_rnn", 10), activation = "tanh", return_sequences=True, input_shape=(config.get("time_steps", 10), 3)),
                attention(),
                #layers.Dropout(0.5),
                layers.Dense(units = 2, activation="linear")
            ]
        )

    def call(self, inputs):
        return self.model(inputs)