import numpy as np
import itertools as iter
#from keras.layers import Input, Dense
#from keras.models import Model
#import tensorflow as tf

# Generating P matrix
# @type numpy.array
print("Start to generate P matrix")

isFirst = True
for p_11, p_12, p_13, p_21, p_22, p_23, p_31, p_32, p_33 in iter.product([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]):
  if isFirst == True:
    _old = np.array([[p_11,p_12,p_13],[p_21, p_22, p_23],[p_31, p_32, p_33]])
    isFirst = False
    continue
  else:
    _new = np.array([[p_11,p_12,p_13],[p_21, p_22, p_23],[p_31, p_32, p_33]])
  
  _old = np.dstack((_old, _new))

print(_old)
# P matrix into Q matrix



#autoencoder = Model(input_img,encoded)