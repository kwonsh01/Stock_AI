import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime

model = keras.models.load_model('samsung.h5')

# x_data = pd.read_csv('x_test.csv')
# y_data = pd.read_csv('y_test.csv')

# y_test = y_data['0'].values
# x_test = x_data.values
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

data = pd.read_csv('samsung.csv')
#print(data.head())

High = data['High'].values
Low = data['Low'].values
Mid = (High + Low)/2

seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(Mid) - sequence_length + 1):
    result.append(Mid[index : index + sequence_length])

Data = []
for window in result:
    normalized_window = [(float(p) / 50000 - 1) for p in window]
    Data.append(normalized_window)

result = np.array(Data)

start = 1500

x_test = result[start:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[start:, -1]

pred = model.predict(x_test)

pred = (pred + 1) * 50000;
y_test = (y_test + 1) * 50000;

# fig = plt.figure(facecolor='white', figsize=(20, 10))
# ax = fig.add_subplot(111)
# ax.plot(y_test, label='True')
# ax.plot(pred, label='Prediction')
# ax.legend()
# plt.show()

future = []
future.append(Mid[len(Mid) - 50 :])
future = np.array(future)
future = (future / 50000) - 1
future = np.reshape(future, (1, seq_len, 1))

predict = model.predict(future)
predict = (predict + 1) * 50000

print("Today: {:.0f}".format(y_test[len(pred) - 1]))
print("Predicted(Today): {:.0f}".format(float(pred[len(pred) - 1])))
print("Predict(Tomorrow): {:.0f}".format(float(predict[len(predict) - 1])))