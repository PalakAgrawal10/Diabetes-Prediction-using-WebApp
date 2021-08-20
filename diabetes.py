import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Diabetes Prediction App
This app predicts that the person is diabetic or **NOT**!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](diabetes.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies',1,17,3)
        Glucose = st.sidebar.slider('Glucose', 0,199,121)
        BloodPressure = st.sidebar.slider('Blood Pressure', 0,122,69)
        SkinThickness = st.sidebar.slider('Skin Thickness', 0,99,21)
        Insulin = st.sidebar.slider('Insulin', 0,846,80)
        BMI = st.sidebar.slider('BMI', 0.0,67.1,32.0)
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.078,2.42,0.47)
        Age = st.sidebar.slider('Age', 21,81,33)
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire diabetes dataset
diabetes_raw = pd.read_csv("diabetes.csv")
diabetes = diabetes_raw.drop(columns=['Outcome'], axis=1)
df = pd.concat([input_df,diabetes],axis=0)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
if st.button("\nPredict"):
    load_clf = pickle.load(open('model_pickle', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)


    st.subheader('Prediction')
    Outcome = np.array([0,1])
    if(Outcome[prediction]==0):
        st.write('No need to worry! Person is Healthy')
    else:
        st.write('Person is Diabetic')
