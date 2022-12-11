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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Constants
DATA_PATH = "data/preprocessed"
CONFIG_PATH = "src/train_params.json"
LOAD_WEIGHTS = False

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


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


def get_dataX(filenames):
    X = np.array([])
    Y = np.array([])
    for filename in filenames:
        df = pd.read_csv(os.path.join(DATA_PATH, filename))
        x, y = get_XY(df)
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
        #xpad = np.zeros((config.get("time_steps"), 3))
        #ypad = np.zeros((config.get("time_steps"), 2)).reshape(-1, 2, 1)
        x = df.to_numpy()[:,2:]
        x[:,2][x[:,2] > 1] = 1
        x[:,1][x[:,1] > 1] = 1
        x[:,0] = [0]+[x[i,0]-x[i-1,0] for i in range(1, len(x))]
        y = x[:,1:]
        #X_full  = np.concatenate((xpad, x))
        X.append(x)
        Y.append(y)
    return X, Y


def calc_metrics(xtrue1, xtrue2, xpred1, xpred2):
    acc1 = accuracy_score(xtrue1, xpred1)
    acc2 = accuracy_score(xtrue2, xpred2)
    prec1 = precision_score(xtrue1, xpred1, average='weighted', zero_division=0)
    prec2 = precision_score(xtrue2, xpred2, average='weighted', zero_division=0)
    recall1 = recall_score(xtrue1, xpred1, average='weighted', zero_division=0)
    recall2 = recall_score(xtrue2, xpred2, average='weighted', zero_division=0)
    return acc1, acc2, prec1, prec2, recall1, recall2


def test_sequence():
    # Prediction
    accuracies1 = []
    accuracies2 = []
    precisions1 = []
    precisions2 = []
    recalls1 = []
    recalls2 = []
    predicted1 = []
    predicted2 = []
    alltrue1 = []
    alltrue2 = []
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
        true1 = list(testY[index][config.get("time_steps"):,0])
        true2 = list(testY[index][config.get("time_steps"):,1])

        pred1 = np.array(predicted_song)[:,0]
        pred2 = np.array(predicted_song)[:,1]
        predicted1.append(pred1)
        predicted2.append(pred2)
        alltrue1.append(true1)
        alltrue2.append(true2)

        acc1, acc2, prec1, prec2, recall1, recall2 = calc_metrics(true1, true2, pred1, pred2)
        print("--------------------------------------------------------")
        print(confusion_matrix(true1, pred1))
        print(confusion_matrix(true2, pred2))

        print(f"Accracy 1: {acc1}")
        print(f"Accuracy 2: {acc2}")
        print(f"Precision 1: {prec1}")
        print(f"Precision 2: {prec2}")
        print(f"Recall 1: {recall1}")
        print(f"Recall 2: {recall2}")

        accuracies1.append(acc1)
        accuracies2.append(acc2)
        precisions1.append(prec1)
        precisions2.append(prec2)
        recalls1.append(recall1)
        recalls2.append(recall2)

    # Final acc:
    mlflow.log_metric("accuracy1", np.mean(accuracies1))
    mlflow.log_metric("accuracy2", np.mean(accuracies2))
    mlflow.log_metric("precision1", np.mean(precisions1))
    mlflow.log_metric("precision2", np.mean(precisions2))
    mlflow.log_metric("recall1", np.mean(recalls1))
    mlflow.log_metric("recall2", np.mean(recalls2))

    predicted1 = [item for sublist in predicted1 for item in sublist]
    predicted2 = [item for sublist in predicted2 for item in sublist]
    alltrue1 = [item for sublist in alltrue1 for item in sublist]
    alltrue2 = [item for sublist in alltrue2 for item in sublist]

    mlflow.log_metric("sum_pred1", np.sum(np.array(predicted1).reshape(-1)))
    mlflow.log_metric("sum_pred2", np.sum(np.array(predicted2).reshape(-1)))

    mlflow.log_metric("sum_true1", np.sum(np.array(alltrue1).reshape(-1)))
    mlflow.log_metric("sum_true2", np.sum(np.array(alltrue2).reshape(-1)))

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
    mlflow.log_param("model_name", "forecast")

    # Tensorboard logs dir
    log_dir = "logs/fit/many2many_simple" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    opt = keras.optimizers.Adam(config.get("learning_rate"))
    model.compile(optimizer=opt, loss='mse', metrics=[keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])

    #if os.path.exists('weights.h5') and LOAD_WEIGHTS:
    #    model.load_weights('weights.h5')

    mlflow.tensorflow.autolog()
    history = model.fit(trainX, trainY, 
                        epochs=config.get("epochs", 10), 
                        validation_split=0.2, 
                        batch_size=config.get("batch_size", 5), 
                        verbose=1, 
                        callbacks=[tensorboard_callback])

    model.save_weights('weights.h5')
    
    test_sequence()
            
