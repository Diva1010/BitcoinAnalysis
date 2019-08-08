#!/usr/bin/env python
# coding: utf-8

from numpy import loadtxt
from keras.models import load_model
import numpy as np

#Load trained model
model = load_model('model.h5')

#import 'X' and 'Y' scalers
from sklearn.externals import joblib
scaler = joblib.load('scaler.save')
scaler1= joblib.load('scaler1.save') 

#X-axis data used to predict the next 'Y'.
X_test=[[9890.05,9668.52,9882.43,31960.31,312717110.04]]

x_test=scaler.transform(X_test)

x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_pred=model.predict(x_test)
y_pred=scaler1.inverse_transform(y_pred)

#print predicted value
y_pred




