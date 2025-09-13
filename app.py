import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

#Load the trained Model
model=tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit App
st.title('Customer Churn Prediction')
#Input fields
geography=st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', min_value=0, max_value=100, value=30)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure', min_value=0, max_value=10, value=5)
num_of_products=st.slider('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])


input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    })

# One Hot Encode 'Geography' feature
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combine the input data with the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling the input data
input_scaled = scaler.transform(input_data)

#Predict Churn Probability
prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]

if st.button('Predict Churn Probability'):
   
    if prediction_probability>0.5:
        st.write('The Customer is LIKELY to Churn')
    else:
        st.write('The Customer is NOT LIKELY to Churn')

