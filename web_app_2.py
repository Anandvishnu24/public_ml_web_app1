# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:04:39 2024

@author: VISHNU
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
loaded_model=pickle.load(open('trained_model (2).sav','rb'))
scaler=pickle.load(open('scaler.sav','rb'))


def diabetes_prediction(input_data):
    input_data_array=np.asarray(input_data)
    input_reshaped=input_data_array.reshape(1,-1)
    sta_data=scaler.transform(input_reshaped)
    prediction=loaded_model.predict(sta_data)
    print(prediction)
    
    if(prediction[0]==0):
        return 'The person is non-diabetic'
    else:
        return ' The person is diabetic'
def  main():
    st.title('Diabetes prediction using machine learning')
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure')
    SkinThickness=st.text_input('Skin Thickness value')
    Insulin=st.text_input('Insuline value')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age')
  
   
    diagnosis=''
    if st.button('Diabetes test reult'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    

    st.success(diagnosis)


if __name__=='__main__':
    main()
    