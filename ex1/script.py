import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

input_data_path="4x4matrix.csv"

p_matrix = np.array(pd.read_csv(input_data_path, index_col=0).iloc[:100,0:16])
q_matrix = np.array(pd.read_csv(input_data_path, index_col=0).iloc[:100,16:])
model = Sequential()
model.add( Dense(16, activation = 'relu', input_dim = 9) )
model.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])

model.fit(q_matrix, p_matrix,epochs = 4)