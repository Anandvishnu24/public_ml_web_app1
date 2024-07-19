# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import streamlit
from sklearn.preprocessing import StandardScaler
scaler=pickle.load(open('B:/ML/scaler.sav','rb'))
loaded_model=pickle.load(open('B:/ML/trained_model (2).sav','rb'))
input=(7,159,66,0,0,30.4,0.383,36)
array=np.asarray(input)
reshaped=array.reshape(1,-1)
std_data=scaler.transform(reshaped)
predict=loaded_model.predict(std_data)
print(predict)


if predict[0]==0:
  print("non diabetic")
else:
  print("diabetic")
  
  