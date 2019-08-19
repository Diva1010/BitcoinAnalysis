#!/usr/bin/env python
# coding: utf-8
import json
import requests
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

sns.set_palette('Set2')
get_ipython().run_line_magic('matplotlib', 'inline')

#Read the data from the file
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
df = pd.DataFrame(json.loads(res.content)['Data'])
df.time = pd.to_datetime(df.time, unit='s')
df.time=df.time.dt.strftime('%Y-%m-%d')
df = df.set_index('time')

#Combine Data into training and testing set
split_row = len(df) - int(0.4 * len(df))
train_data = df.iloc[:split_row]
test_data = df.iloc[split_row:]

ind=df[1202:2001].index.tolist()

#plot test data vs train data graph
fig, ax = plt.subplots(1, figsize=(16, 9))
ax.plot(train_data['close'], label='training', linewidth=2)
ax.plot(test_data['close'], label='test', linewidth=2)

ax.set_ylabel('price [USD]', fontsize=14)
ax.set_title('BTC', fontsize=18)
ax.legend(loc='best', fontsize=18);

#seperate 'x' axis and 'y' axis for train and test data
x_train = train_data[['high','low','open','volumefrom','volumeto']]
y_train = train_data[['close']]
x_test = test_data[['high','low','open','volumefrom','volumeto']]
y_test = test_data[['close']]


#Scale data to normalize it
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
y_train = scaler1.fit_transform(y_train)
x_test = scaler.transform(x_test)
y_test = scaler1.transform(y_test)

#save 'X' scaler to a file
from sklearn.externals import joblib
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 

#Save 'Y' scaler to a file
scaler_filename = "scaler1.save"
joblib.dump(scaler1, scaler_filename) 

#Re-shape 'x' to use it for training
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

#Train the Model
model = Sequential()
model.add(LSTM(20, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('linear'))
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mae', optimizer=adam)
history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=1, shuffle=True)

#Save the trained model to a file to use it later
model.save('model.h5')


#Find predicted value and check for Mean-squared error
preds = model.predict(x_test).squeeze()
mean_absolute_error(preds, y_test)

#Inverse Transform 'Y' to plot on graph
y_inv=scaler1.inverse_transform(y_test).tolist()
y_list = [item for sublist in y_inv for item in sublist]
pred_lst=scaler1.inverse_transform([preds])[0].tolist()

#Plot Actual VS Predicted price
fig, ax = plt.subplots(1, figsize=(16, 9))
ax.plot(ind[0:50],pred_lst[0:50],label='predictions')
ax.plot(ind[0:50],y_list[0:50],label='actual')
ax.legend(loc='best', fontsize=18)
plt.xticks(rotation=90)
plt.show()




