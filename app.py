import streamlit as st

import os
import pickle
import pandas as pd

from tensorflow.keras.models import load_model

gender_transformation_path = os.path.join('feature_transformation_artifacts', 'gender_ohe.pkl')
geography_transformation_path = os.path.join('feature_transformation_artifacts', 'geography_ohe.pkl')
scaler_path = os.path.join('feature_transformation_artifacts', 'standard_scaler.pkl')
model_path = os.path.join('model_artifacts', 'customer_churn_model.keras')

with open(gender_transformation_path, 'rb') as f:
    gender_ohe = pickle.load(f)

with open(geography_transformation_path, 'rb') as f:
    geography_ohe = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

model = load_model(model_path)

## streamlit app
st.title('Customer Churn PRediction')


# User input
geography = st.selectbox('Geography', geography_ohe.categories_[0])
gender = st.selectbox('Gender', gender_ohe.categories_[0])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
print(input_df)

geography_encoded = geography_ohe.transform([[geography]])
geography_df = pd.DataFrame(geography_encoded, columns=geography_ohe.get_feature_names_out())

gender_encoded = gender_ohe.transform([[gender]])
gender_df = pd.DataFrame(gender_encoded, columns=gender_ohe.get_feature_names_out())


final_df = pd.concat([input_df, geography_df, gender_df], axis=1, ignore_index=False)
print(final_df)

print(final_df.to_dict())

scaled_data = scaler.transform(final_df)
print(scaled_data)

if st.button('Predict Churn'):
    prediction = model.predict(scaled_data)
    churn_probability = prediction[0][0]
    churn_prediction = 'Yes' if churn_probability > 0.5 else 'No'
    
    st.write(f'Churn Prediction: {churn_prediction}')
    st.write(f'Churn Probability: {churn_probability:.2f}')