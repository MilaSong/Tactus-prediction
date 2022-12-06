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


def get_XY(durations, tactus, joint, time_steps):
    for tid in range(1, len(tactus)):
        durations[tid] = durations[tid] - durations[tid-1]
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


def get_dataX(filenames):
    X = np.array([])
    Y = np.array([])
    for filename in filenames:
        df = pd.read_csv(os.path.join(DATA_PATH, filename))
        x, y = get_XY(np.array(df.duration), np.array(df.tactus), np.array(df.joint), config.get("time_steps", 10))
        if len(X) < 1:
            X = x
            Y = y
        else:
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))
    return X, Y


def get_dataY(filenames):
    X = []
    Y = []
    for filename in filenames:
        df = pd.read_csv(os.path.join(DATA_PATH, filename))
        xpad = np.zeros((config.get("time_steps"), 3))
        ypad = np.zeros((config.get("time_steps"), 2)).reshape(-1, 2, 1)
        x = df.to_numpy()[:,2:]
        y = df.to_numpy()[:,3:]
        X_full  = np.concatenate((xpad, x))
        X.append(X_full)
        Y.append(y)
    return X, Y


def calc_metrics(xtrue1, xtrue2, xpred1, xpred2):
    acc1 = accuracy_score(xtrue1, xpred1)
    acc2 = accuracy_score(xtrue2, xpred2)

    prec1 = precision_score(xtrue1, xpred1)
    prec2 = precision_score(xtrue2, xpred2)

    recall1 = recall_score(xtrue1, xpred1)
    recall2 = recall_score(xtrue2, xpred2)

    return acc1, acc2, prec1, prec2, recall1, recall2



model = LSTMModel()
model.build(input_shape=(1, config.get("time_steps", 10), 3))
plot_model(model.model, to_file=f"resources/{model.model_name.lower()}_structure.png", show_shapes=True, show_layer_names=True)

with mlflow.start_run() as run:
    run_id = run.info.run_id

    filenames = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    filenames_train, filenames_test = train_test_split(filenames, test_size=0.1, random_state=42)
    trainX, trainY = get_dataX(filenames_train)
    testX, testY = get_dataY(filenames_test)

    mlflow.log_param("epochs", config.get("epochs"))
    mlflow.log_param("batch_size", config.get("batch_size"))
    mlflow.log_param("time_steps", config.get("time_steps"))
    mlflow.log_param("hidden_lstm", config.get("hidden_lstm"))
    mlflow.log_param("model_name", model.model_name)

    # Tensorboard logs dir
    log_dir = "logs/fit/many2many_simple" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    opt = keras.optimizers.Adam(config.get("learning_rate"), 1e-3)
    model.compile(optimizer=opt, loss='mse', metrics=[keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])
    mlflow.tensorflow.autolog()
    history = model.fit(trainX, trainY, 
                        epochs=config.get("epochs", 10), 
                        validation_split=0.2, 
                        batch_size=config.get("batch_size", 5), 
                        verbose=1, 
                        callbacks=[tensorboard_callback])

    # Prediction
    accuracies1 = []
    accuracies2 = []
    for index, testsong in enumerate(testX):
        predicted_song = []
        x_test = testsong[:config.get("time_steps")]
        for i in range(config.get("time_steps"), len(testsong)):
            x_test = np.asarray(x_test).astype('float32').reshape(-1, config.get("time_steps"), 3)
            predicted = model.predict(x_test, verbose=0).round()
            next_time = testsong[i][0]
            x_next_predicted = np.concatenate((np.array(next_time).reshape(-1), predicted.reshape(-1)))
            predicted_song.append(list(x_next_predicted[1:]))
            x_test = np.concatenate((x_test[:, 1:].reshape(-1,3), x_next_predicted.reshape(-1,3))).reshape(1, -1, 3)

        # Metrics
        true1 = list(testY[index][:,0])
        true2 = list(testY[index][:,1])

        pred1 = np.array(predicted_song)[:,0]
        pred2 = np.array(predicted_song)[:,1]

        acc1, acc2, prec1, prec2, recall1, recall2 = calc_metrics(true1, true2, pred1, pred2)

        print(f"Accracy 1: {acc1}")
        print(f"Accuracy 2: {acc2}")
        print(f"Precision 1: {prec1}")
        print(f"Precision 2: {prec2}")
        print(f"Recall 1: {recall1}")
        print(f"Recall 2: {recall2}")

        accuracies1.append(acc1)
        accuracies2.append(acc2)
        break


# Final acc:
print("-----------------------------------")
print(f"Mean accuracy1 = {np.mean(accuracies1)}")
print(f"Mean accuracy2 = {np.mean(accuracies2)}")
            

