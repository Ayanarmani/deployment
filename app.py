import numpy as np
import pickle
import streamlit as st

#Loading the saved model
loaded_model = pickle.load(open('trained-model.pkl','rb'))

#Create a function for prediction
def heart_disease_prediction(input_data):

    #changing the input into numpy array
    input_data_as_numpy_array = np.array(input_data, dtype=float)

    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "This Person don't have heart disease"
    else:
        return "This Person has heart disease! Better consult with doctor"

def main():

    # giving title
    st.title("Heart Disease Prediction")

    #getting the input data from the user
    age = st.text_input("Age")
    sex = st.text_input("Gender (1-male: 0-female)")
    cp = st.text_input("Chest Pain (eg: 0-Typical angina, 1-Atypical angina, 2-Non-anginal, 3-Asymptomatic)")
    fbs = st.text_input("Fasting blood sugar > 120mg/dl (eg:0-False, 1-True)")
    restecg = st.text_input("Resting ECG result (eg: 0-nothing, 1:ST-T Abnormality, 2:Possible Hypertrophy)")
    exang = st.text_input("Exercise induced angina (eg: 1=yes; 0 = no)")
    oldpeak = st.text_input("ST depression value")
    slope = st.text_input("Slopping (eg:0-upslopping, 1-flatslopping, 2-downslopping)")
    ca = st.text_input("No.of Vessels coloured by flourosopy (eg:0-3")
    thal = st.text_input("Thalium stress level (eg: 1-3:Normal, 6-fixed defect, 7-reversable defect)")

    #code for prediction
    diagnosis = ""

    #creating a button for prediction
    if st.button("Heart Disease Test result"):
        diagnosis = heart_disease_prediction([age,sex,cp,fbs,restecg,exang,oldpeak,slope,ca,thal])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
