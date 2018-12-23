import numpy as np
import itertools as iter
#from keras.layers import Input, Dense
#from keras.models import Model
#import tensorflow as tf

# @return P matrix
def generatePMatrix():
  _temp = np.zeros((2 ** 9,3,3))
  _count = 0
  for p_11, p_12, p_13, p_21, p_22, p_23, p_31, p_32, p_33 in iter.product([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]):
    _temp[_count] = np.array([[p_11, p_12, p_13],[p_21, p_22, p_23],[p_31, p_32, p_33]]) 
    _count += 1
  
  return _temp

# Generating P matrix
# @type numpy.array
print("Start to generate P matrix")
P_matrix = generatePMatrix()
print("Generated P matrix shaped " + str(P_matrix.shape))


# P matrix into Q matrix



#autoencoder = Model(input_img,encoded)