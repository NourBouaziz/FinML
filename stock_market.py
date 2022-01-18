import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

stocks = pd.read_csv('stocks_to_buy.csv', thousands=',')

stocks = stocks.dropna(subset=['Date', 'Open', 'Daily_High', 'Daily_Low', 'Volume', 'Closing_Price', 'Volume'])
stocks = stocks[stocks['Name'] == 'Accor']
stocks = stocks.reset_index()
stocks = stocks.drop(columns=['index', 'Name'])
stocks = stocks[::-1]
stocks = stocks[-500:]

# setting output var and features
output_var = pd.DataFrame(stocks['Closing_Price'])
features = ['Open', 'Daily_High', 'Daily_Low', 'Volume']

# scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(stocks[features])

# splitting data
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[
                                                            len(train_index): (len(train_index) + len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (
            len(train_index) + len(test_index))].values.ravel()
    print('test', test_index)
    print('train', train_index)

# processing data
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

X_train = np.squeeze(X_train)
X_test = np.squeeze(X_test)

# initializing
m = 16  # number of rules
alpha = 0.01  # learning rate
D = 4  # number of features

fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)

# Training
num_epochs = 20000

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):

        trn_loss, trn_pred = fis.train(sess, X_train, y_train)

        val_pred, val_loss = fis.infer(sess, X_test, y_test)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
            print("Validation loss: %f" % val_loss)
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            # Plot real vs. predicted
            pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            plt.figure('Stock for Accor')
            plt.plot(output_var)
            plt.plot(pred)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)

    plt.figure('Loss')
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(trn_costs))
    plt.title("Training loss, Learning rate =" + str(alpha))
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(val_costs))
    plt.title("Validation loss, Learning rate =" + str(alpha))
    plt.ylabel('Cost')
    plt.xlabel('Epochs')

    plt.show()
