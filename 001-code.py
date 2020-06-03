import tensorflow
import keras
import pandas
import numpy
import os
import matplotlib.pyplot as plt

#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = './week_8'

tv_sales = pandas.read_csv(os.path.join(dir_path, 'tv_sales.csv'))

# Firstly we need to pre-process the data, convert all strings to numbers
# First thing we must convert the tv models to a number
df = tv_sales.copy()
# Get all the tv models in the system
tv_models_lst = list(df.Model.unique())
# ensure we have a reproductable value in the future
tv_models_lst.sort()
# Create the set of the data to use the map function
tv_models = { tv_models_lst[idx]:idx for idx in range(len(tv_models_lst))}

df.Model = df.Model.map(tv_models)


# Reverse the array
df = df[::-1]

# Convert the date to a number
df.Date = pandas.to_datetime(df.Date).astype(int)
dates = df.Date.unique()
dates.sort()
dates_map = { dates[idx]: idx for idx in range(len(dates)) }
df.Date = df.Date.map(dates_map)

# Standardize all the numbers to be a value between 0-1
# This is a simple normalization process, no need to use a third party function
df.Count = (df.Count - df.Count.min()) / (df.Count.max() - df.Count.min())
df.Model = (df.Model - df.Model.min()) / (df.Model.max() - df.Model.min())
df.Date = (df.Date - df.Date.min()) / (df.Date.max() - df.Date.min())

# Split the data into windows of 30 days
window_size = 30

# Split the data into test and train data. In this case lets save the last window size + 5 days to test
train_data = df[:len(df)-window_size-5]
test_data = df[len(df)-window_size-5:]

x_train = []
y_train = []
for i in range(window_size, len(train_data)):
    x_train.append(train_data.iloc[i-window_size:i, 0:].values) # Training = date only
    y_train.append(train_data.iloc[i, -1])

x_train, y_train = numpy.array(x_train), numpy.array(y_train).reshape(-1, 1)

x_test = []
y_test = []
for i in range(window_size, len(test_data)):
    x_test.append(test_data.iloc[i-window_size:i, 0:].values) # Training = date only
    y_test.append(test_data.iloc[i, -1])

x_test, y_test = numpy.array(x_test), numpy.array(y_test).reshape(-1, 1)

model_types = [
    lambda s: s.add(keras.layers.SimpleRNN(units=100, input_shape=(window_size, x_train.shape[2]), name='RNN')),
    # lambda s: s.add(keras.layers.LSTM(units=100, input_shape=(window_size, x_train.shape[2]), name='LSTM')),
    # lambda s: s.add(keras.layers.GRU(units=100, input_shape=(window_size, x_train.shape[2]), name='RNN')),
]

for model in model_types:
    sequence = keras.models.Sequential()
    model(sequence)
    sequence.add(keras.layers.Dropout(.5, name='dropout'))
    sequence.add(keras.layers.Dense(units=10, name='dense'))
    sequence.add(keras.layers.Dense(units=1, name='output'))
    sequence.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mae')
    model_history=sequence.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test))

plt.plot(model_history.history['loss'], label='train')
plt.plot(model_history.history['val_loss'], label='test')
plt.show()

train_pred=sequence.predict(x_train)
err = np.mean(np.abs(y_train-train_pred))
print('train MAE for standard averaging: ', err)

test_pred=sequence.predict(x_test)
err = np.mean(np.abs(y_test-test_pred))
print('test MAE for standard averaging: ', err)

plt.figure(figsize=(18,6))
N_train=x_train.shape[0]
N_test=x_test.shape[0]
plt.plot(y_test, color='b', label='true')
plt.plot(test_pred,color='orange', label='predicted')
plt.show()