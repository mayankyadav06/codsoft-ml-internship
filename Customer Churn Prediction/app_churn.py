import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('E:\Codsoft\Customer Churn Prediction\churn_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

def convert_inputs(geography, gender):
    data = {
        'Geography_France': 0,
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Female': 0,
        'Gender_Male': 0
    }
    data[f'Geography_{geography}'] = 1
    data[f'Gender_{gender}'] = 1
    return data

st.title('Customer Churn Prediction')

# Define your inputs
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=500)
age = st.number_input('Age', min_value=0, max_value=100, value=25)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.00, max_value=1000000.00, value=50000.00)
num_of_products = st.number_input('Number of Products', min_value=0, max_value=10, value=1)
has_cr_card = st.number_input('Has Credit Card', min_value=0, max_value=1, value=0)
is_active_member = st.number_input('Is Active Member', min_value=0, max_value=1, value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.00, max_value=1000000.00, value=50000.00)
geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))
gender = st.selectbox('Gender', ('Female', 'Male'))

if st.button('Predict'):
    # Convert inputs to binary values
    binary_inputs = convert_inputs(geography, gender)

    # Create a dataframe from the inputs
    data = {'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card, 'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary}
    data.update(binary_inputs)
    df = pd.DataFrame([data])

    # Apply standard scaler
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    # Make prediction
    prediction = model.predict(df)

    # Display prediction
    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is likely to stay.')
