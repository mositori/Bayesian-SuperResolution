import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter
import sqlite3
from progressbar import ProgressBar

def generatePMatrix():
  _temp = np.zeros((2 ** 16,4,4))
  _count = 0
  for p_11, p_12, p_13, p_14, p_21, p_22, p_23, p_24, p_31, p_32, p_33, p_34, p_41, p_42, p_43, p_44 in iter.product([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]):
    _temp[_count] = np.array([[p_11, p_12, p_13, p_14],[p_21, p_22, p_23, p_24],[p_31, p_32, p_33, p_34],[p_41, p_42, p_43, p_44]]) 
    _count += 1
  
  return _temp


def calcQMatrix(data, p_size=(4,4),a_size=(2,2)):
    q_size = (p_size[0]-a_size[0]+1, p_size[1]-a_size[1]+1)
    _columns = []
    p = ProgressBar(2**20)
    for _y in range(1, p_size[1]+1):
        for _x in range(1, p_size[0]+1):
            _index = "p_{y}{x}".format(x = _x, y = _y)
            _columns.append(_index)
    
    for _y in range(1, q_size[1]+1):
        for _x in range(1, q_size[0]+1):
            _index = "q_{y}{x}".format(x = _x, y = _y)
            _columns.append(_index)
            
    
    df = pd.DataFrame(columns=_columns)
    _name_count = 0
    
    for i in range( data.shape[0] ):
        _data = data[i]
        for a_11, a_12, a_21, a_22 in iter.product([0,1],[0,1],[0,1],[0,1]):
            q_11 = _data[0][0] * a_11 + _data[0][1] * a_12 + _data[1][0] * a_21 + _data[1][1] * a_22
            q_12 = _data[0][1] * a_11 + _data[0][2] * a_12 + _data[1][1] * a_21 + _data[1][2] * a_22
            q_13 = _data[0][2] * a_11 + _data[0][3] * a_12 + _data[1][2] * a_21 + _data[1][3] * a_22
            
            q_21 = _data[1][0] * a_11 + _data[1][1] * a_12 + _data[2][0] * a_21 + _data[2][1] * a_22
            q_22 = _data[1][1] * a_11 + _data[1][2] * a_12 + _data[2][1] * a_21 + _data[2][2] * a_22
            q_23 = _data[1][2] * a_11 + _data[1][3] * a_12 + _data[2][2] * a_21 + _data[2][3] * a_22
            
            q_31 = _data[2][0] * a_11 + _data[2][1] * a_12 + _data[3][0] * a_21 + _data[3][1] * a_22
            q_32 = _data[2][1] * a_11 + _data[2][2] * a_12 + _data[3][1] * a_21 + _data[3][2] * a_22
            q_33 = _data[2][2] * a_11 + _data[2][3] * a_12 + _data[3][2] * a_21 + _data[3][3] * a_22
            
            _insert = pd.Series([_data[0][0], _data[0][1], _data[0][2], _data[0][3],
                                 _data[1][0], _data[1][1], _data[1][2], _data[1][3],
                                 _data[2][0], _data[2][1], _data[2][2], _data[2][3],
                                 _data[3][0], _data[3][1], _data[3][2], _data[3][3],
                                 q_11, q_12, q_13,
                                 q_21, q_22, q_23,
                                 q_31, q_32, q_33], index=df.columns, name = _name_count)
            df = df.append(_insert)
            _name_count += 1
            p.update(_name_count + 1)
    
    df.to_csv("4x4matrix.csv")
    df.head(5)
    df.tail(5)
    return df
#p_matrix = generatePMatrix()
#df = calcQMatrix(p_matrix)

df = pd.read_csv("../4x4matrix.csv")
