
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("loan.csv")

df.dropna(subset=['Gender'], inplace=True)

def cap_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[column].clip(lower_bound, upper_bound)

df['ApplicantIncome'] = cap_outliers('ApplicantIncome')
df['CoapplicantIncome'] = cap_outliers('CoapplicantIncome')
df['LoanAmount'] = cap_outliers('LoanAmount')

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

df.drop(columns=['Loan_ID'], axis=1, inplace=True)

df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
df['Married'].replace(['No', 'Yes'], [0, 1], inplace=True)
df['Education'].replace(['Graduate', 'Not Graduate'], [0, 1], inplace=True)
df['Self_Employed'].replace(['No', 'Yes'], [0, 1], inplace=True)
df['Dependents'].replace(['0', '1', '2', '3+'], [0, 1, 2, 3], inplace=True)

df = pd.get_dummies(df, columns=['Property_Area'])

df["Loan_Status"].replace(['Y', 'N'], [1, 0], inplace=True)
df["Property_Area_Rural"].replace([True, False], [1, 0], inplace=True)
df['Property_Area_Semiurban'].replace([True, False], [1, 0], inplace=True)
df['Property_Area_Urban'].replace([True, False], [1, 0], inplace=True)


x = df.drop(columns=['Loan_Status'])
x.fillna(0, inplace=True)
y = df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel='linear', C=5.0, gamma='scale', degree=3)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

st.title("Loan Status Prediction")

st.success(f"Model Accuracy: {accuracy:.2f}")

gender = st.selectbox("Gender", [0, 1])
married = st.selectbox("Married", [0, 1])
dependents = st.number_input("Dependents", min_value=0, value=0)
education = st.selectbox("Education", [0, 1])
self_employed = st.selectbox("Self Employed", [0, 1])
applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=1000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
credit_history = st.selectbox("Credit History", [0.0, 1.0])

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

property_rural = 1 if property_area == "Rural" else 0
property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0

if st.button("Predict"):

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area_Rural': [property_rural],
        'Property_Area_Semiurban': [property_semiurban],
        'Property_Area_Urban': [property_urban]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
