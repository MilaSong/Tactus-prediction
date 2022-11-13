import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
import datetime
import os


# Constants
DATA_PATH = "data/preprocessed"
TIME_STEPS = 10



def get_XY(durations, tactus, joint, time_steps):

    j = [np.arange(0, x+1) for x in np.arange(len(durations))[:(time_steps-1)]]
    k = [np.arange(x, x + time_steps) for x in np.arange(len(durations))[:-(time_steps)]]

    X = []
    for i, arr in enumerate(j + k):
        arr2 = np.vstack([durations[arr], tactus[arr], joint[arr]]).T
        arr2 = np.pad(arr2, ((time_steps-len(arr), 0), (0,0)))
        X.append(arr2)
    
    X = np.array(X)
    Y_ind = np.arange(1, len(durations))
    Y = np.vstack((tactus[Y_ind], joint[Y_ind])).T
    Y = Y.reshape(-1, 2, 1)

    return X, Y


X = np.array([])
Y = np.array([])
filenames = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
for filename in filenames:
    df = pd.read_csv(os.path.join(DATA_PATH, filename))
    x, y = get_XY(np.array(df.duration), np.array(df.tactus), np.array(df.joint), TIME_STEPS)
    if len(X) < 1:
        X = x
        Y = y
    else:
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))

    # Concatenate all the other files
    # ...


model = keras.Sequential( 
    [
        layers.LSTM(units = 20, activation = "relu", input_shape=(TIME_STEPS, 3)),
        layers.Dense(units = 1, activation = "linear"),
        #layers.Dropout(0.5),
        layers.RepeatVector(2),
        layers.LSTM(units = 20, activation = "relu", return_sequences=True),
        #layers.Dropout(0.5),
        layers.TimeDistributed(layers.Dense(units = 1, activation="sigmoid"))
    ]
)

# Tensorboard logs dir
log_dir = "logs/fit/many2many_simple" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=10, validation_split=0.2, batch_size=10, verbose=1, callbacks=[tensorboard_callback])


