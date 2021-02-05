# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:17:15 2021

@author: Admin
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)


df = pd.read_csv(r'C:\Users\Admin\Downloads\BreastCancer.csv')
df


dum_df = pd.get_dummies(df)
dum_df.drop(['Class_Benign'],axis="columns",inplace = True)
dum_df.head()

X = dum_df.iloc[:,1:10]
y = dum_df.iloc[:,10]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2019,stratify=y)

tf.random.set_random_seed(99)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


inputs = keras.Input(shape=(9,))
x = tf.keras.layers.Dense(6, activation="relu")(inputs)
x = tf.keras.layers.Dense(4, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])


y_train = y_train.values
y_test = y_test.values


history = model.fit( X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=500)


print(model.summary())


%matplotlib inline

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


from sklearn.metrics import log_loss
y_pred_prob = model.predict(X_test)
log_loss(y_true=y_test,y_pred=y_pred_prob)


from sklearn.metrics import accuracy_score
predict_prob = model.predict(X_test)


def toYN(x):
    if x >= 0.5:
        return 1
    else:
        return 0

vf_YN = np.vectorize(toYN)
predict_classes = vf_YN(predict_prob)


acc = accuracy_score(y_test,predict_classes)
print(f"Accuracy: {acc}")