import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import callbacks


# Load the dataset and preprocess
column_names = ['date', 'tickets']
df = pd.read_csv('data_daily.csv', names=column_names)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Sort the DataFrame by date
df.sort_values('date', inplace=True)

# Choose the target variable and features
target_variable = 'tickets'
features = ['day_of_week', 'month']

# Normalize the data manually
min_value = df[target_variable].min()
max_value = df[target_variable].max()
df[target_variable] = (df[target_variable] - min_value) / (max_value - min_value)

# Define a function to create input sequences for the LSTM model
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length][features]
        target = data.iloc[i+sequence_length][target_variable]
        sequences.append(sequence.values)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Choose the sequence length (number of time steps to look back)
sequence_length = 10

# Create sequences and targets
X, y = create_sequences(df, sequence_length)


# split train , validation, test data (70:10:20)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.8)
# X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]
# y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]


#internal cross validation and network parameter tuning

# neurons = [1, 5, 10, 30, 50]
# batch_sizes = [1, 5, 10, 20, 32]
# # Reshape the input for LSTM (number of data points, number of time steps, number of features)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features))) #
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))
# X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(features)))
#
# print(X_val)
#
# loss_vals = []
# # parameter tuning for number of neurons and batch size in LSTM
# for i in neurons:
#     for j in batch_sizes:
#         # Build the LSTM model
#         model = Sequential()
#         sequence_length = 10
#         callback = callbacks.EarlyStopping(monitor='loss', patience=3)
#         model.add(LSTM(i, activation='relu', input_shape=(sequence_length, len(features)))) #RNN LSTM layer
#         model.add(Dense(1)) #output layer
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(X_train, y_train, epochs=500, batch_size=j, callbacks= [callback])
#         test_loss = model.evaluate(X_val, y_val)
#         loss_vals.append((i,j, test_loss))
#
# print(loss_vals)
#
# loss_vals.sort(key=lambda x:x[2])
# neuron, batch_size, loss = loss_vals[0]
# print(f"network parameters after internal cross validation: neuron: {neuron}, batch_size: {batch_size}, loss: {loss}")
neuron = 50
batch_size = 1

# combining the training and validation set

X_train, X_test = X[:val_size], X[val_size:]
y_train, y_test = y[:val_size], y[val_size:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features)))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))

# Build the LSTM model
model = Sequential()
sequence_length = 10
callback = callbacks.EarlyStopping(monitor='loss', patience=3)
model.add(LSTM(neuron, activation='relu', input_shape=(sequence_length, len(features)))) #RNN LSTM layer
model.add(Dense(1)) #output layer
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=500, batch_size=batch_size, callbacks= [callback])
test_loss = model.evaluate(X_test, y_test)
model.save('receipt_count.h5')
print(f"final model loss value: {test_loss}")

y_pred = model.predict(X_test)

# Invert the scaling to get the actual receipt numbers
y_pred_actual = y_pred * (max_value - min_value) + min_value
y_test_actual = y_test * (max_value - min_value) + min_value

print(y_pred_actual)
print(y_test_actual)
