import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import datetime
import os
import json
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import tensorflow as tf
from bilstm import BiLSTMModel
from lstm import LSTMModel
from lstm_attention import LSTMAttModel
from rnn_attention import RNNAttModel
from rnn import RNNModel
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Constants
DATA_PATH = "data/preprocessed"
CONFIG_PATH = "src/train_params.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


def _get_XY(durations, tactus, joint, time_steps):

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

def get_XY(df):
    time_steps = config.get("time_steps", 10)
    data = df[['duration', 'tactus', 'joint']].to_numpy()
    data[:,2][data[:,2] > 1] = 1
    data[:,1][data[:,1] > 1] = 1
    # cut into time since last
    data[:,0] = [0]+[data[i,0]-data[i-1,0] for i in range(1, len(data))]

    # add padding
    data = [0] + np.concatenate((np.zeros((time_steps, 3)), data))

    batch_data_X = []
    batch_data_Y = []
    # binnify
    for i in range(time_steps, len(data)):
        batch_data_X.append(np.array(data[i-time_steps:i,:], dtype=float).reshape((config.get("time_steps"), 3)))
        batch_data_Y.append(np.array(data[i,1:], dtype=float).reshape(2))

    return (np.array(batch_data_X), np.array(batch_data_Y))



def get_data(filenames):
    X = np.array([])
    Y = np.array([])
    for filename in filenames:
        df = pd.read_csv(os.path.join(DATA_PATH, filename))
        #x, y = get_XY(np.array(df.duration), np.array(df.tactus), np.array(df.joint), config.get("time_steps", 10))
        x, y = get_XY(df)
        if len(X) < 1:
            X = x
            Y = y
        else:
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))
    return X, Y


def calc_metrics(xtrue1, xtrue2, xpred1, xpred2):
    acc1 = accuracy_score(xtrue1, xpred1)
    acc2 = accuracy_score(xtrue2, xpred2)
    prec1 = precision_score(xtrue1, xpred1, average='weighted', zero_division=0)
    prec2 = precision_score(xtrue2, xpred2, average='weighted', zero_division=0)
    recall1 = recall_score(xtrue1, xpred1, average='weighted', zero_division=0)
    recall2 = recall_score(xtrue2, xpred2, average='weighted', zero_division=0)
    return acc1, acc2, prec1, prec2, recall1, recall2


model = LSTMModel()
model.build(input_shape=(1, config.get("time_steps", 10), 3))
plot_model(model.model, to_file=f"resources/{model.model_name.lower()}_structure.png", show_shapes=True, show_layer_names=True)

with mlflow.start_run() as run:
    run_id = run.info.run_id

    filenames = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    filenames_train, filenames_test = train_test_split(filenames, test_size=0.2, random_state=42)
    trainX, trainY = get_data(filenames_train)
    testX, testY = get_data(filenames_test)

    mlflow.log_param("epochs", config.get("epochs"))
    mlflow.log_param("batch_size", config.get("batch_size"))
    mlflow.log_param("time_steps", config.get("time_steps"))
    mlflow.log_param("hidden_lstm", config.get("hidden_lstm"))
    mlflow.log_param("model_name", "sigmoidLSTM")

    # Tensorboard logs dir
    log_dir = "logs/fit/many2many_simple" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    opt = keras.optimizers.Adam(config.get("learning_rate"))
    model.compile(optimizer=opt, loss='mse', metrics=[keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])
    mlflow.tensorflow.autolog()
    history = model.fit(trainX, trainY, 
                        epochs=config.get("epochs", 10), 
                        validation_split=0.2, 
                        batch_size=config.get("batch_size", 5), 
                        verbose=1, 
                        callbacks=[tensorboard_callback])

    score = model.evaluate(testX, testY, verbose=0)
    mlflow.log_metric("test_mse", score[0])
    mlflow.log_metric("test_rmse", score[1])
    mlflow.log_metric("test_mae", score[2])

    # Prediction
    predicted = model.predict(testX).round().reshape(-1, 2)

    predicted1 = predicted[:, 0]
    predicted2 = predicted[:, 1]

    true1 = testY[:, 0]
    true2 = testY[:, 1]

    acc1, acc2, prec1, prec2, recall1, recall2 = calc_metrics(true1, true2, predicted1, predicted2)

    print(f"Accracy 1: {acc1}")
    print(f"Accuracy 2: {acc2}")
    print(f"Precision 1: {prec1}")
    print(f"Precision 2: {prec2}")
    print(f"Recall 1: {recall1}")
    print(f"Recall 2: {recall2}")

    mlflow.log_metric("accuracy1", acc1)
    mlflow.log_metric("accuracy2", acc2)
    mlflow.log_metric("precision1", prec1)
    mlflow.log_metric("precision2", prec2)
    mlflow.log_metric("recall1", recall1)
    mlflow.log_metric("recall2", recall2)

    mlflow.log_metric("sum_pred1", sum(predicted1))
    mlflow.log_metric("sum_pred2", sum(predicted2))

    mlflow.log_metric("sum_true1", sum(true1))
    mlflow.log_metric("sum_true2", sum(true2))

    

