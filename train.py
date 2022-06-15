import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime

data = pd.read_csv('samsung.csv')
#print(data.head())

High = data['High'].values
Low = data['Low'].values
Mid = (High + Low)/2

seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(Mid) - sequence_length):
    result.append(Mid[index : index + sequence_length])

Data = []
for window in result:
    normalized_window = [(float(p) / 50000 - 1) for p in window]
    Data.append(normalized_window)

result = np.array(Data)

row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

model = Sequential()
model.add(LSTM(seq_len, return_sequences=True, input_shape=(seq_len, 1)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=25)
model.save('samsung.h5')

# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
# df = pd.DataFrame(x_test)
# df.to_csv('x_test.csv', index=False)
# df = pd.DataFrame(y_test)
# df.to_csv('y_test.csv', index=False)