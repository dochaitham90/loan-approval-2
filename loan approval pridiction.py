import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb


st.set_page_config(layout='wide')
df = pd.read_csv('Loan approval prediction.csv', encoding="ISO-8859-1")
st.write(df.head())
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])
if option == 'Home':
    st.title('Loan Approval Prediction APP')
    st.text('Author: @HAITHAM')
    st.dataframe(df.head(20))
elif option == 'EDA':
    st.title('Exploratory Data Analysis')
    st.text('Author: @HAITHAM')
    
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='loan_status', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='person_home_ownership', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='loan_status', hue='loan_grade', ax=ax)
    st.pyplot(fig)

    st.subheader("Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='loan_amnt', y='loan_percent_income', hue='loan_status')
    st.pyplot(fig)
    
    
    
elif option == "ML":
    st.title('loan Approval Prediction')
    st.text('Author: @HAITHAM')
    st.text("Please enter the following values:")
    age = st.number_input("Age", min_value=0, max_value=100)
    person_income = st.number_input("Person Income", min_value=0, max_value=1000000)
    person_emp_length = st.number_input("Person Employment Length", min_value=0, max_value=50)
    loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=1000000)
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=100.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=100.0)
    cb_person_default_on_file = st.selectbox("cb_person_default_on_file", ["Y", "N"])
    if cb_person_default_on_file == "N":
        cb_person_default_on_file = 0
    else :
        cb_person_default_on_file = 1
    
    cb_person_cred_hist_length = st.number_input("cb_person_default_on_file", min_value=0, max_value=50)
    person_home_ownership = st.selectbox("Person Home Ownership", ["RENT", "MORTGAGE", "OWN","OTHER"])
    if person_home_ownership == "RENT":
        person_home_ownership = 3
    elif person_home_ownership == "MORTGAGE":
        person_home_ownership = 0

    elif person_home_ownership == "OWN":
        person_home_ownership = 2
    else:
        person_home_ownership = 1

    
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
    if loan_intent == "PERSONAL":
        loan_intent = 4
    elif loan_intent == "EDUCATION":
        loan_intent = 1

    elif loan_intent == "MEDICAL":
        loan_intent = 3
    elif loan_intent == "VENTURE":
        loan_intent = 5
    elif loan_intent == "DEBTCONSOLIDATION":
        loan_intent = 0
    else:
        loan_intent = 2

    
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    if loan_grade == "A":
        loan_grade = 0
    elif loan_grade == "B":
        loan_grade = 1
    elif loan_grade == "C":
        loan_grade = 2
    elif loan_grade == "D":
        loan_grade = 3
    elif loan_grade == "E":
        loan_grade = 4

    elif loan_grade == "F":
        loan_grade = 5
    else:
        loan_grade = 6
    
    
    btn = st.button("Predict")
   
    import pandas as pd
    data = {'age': [age], 'person_income': [person_income], 'person_emp_length': [person_emp_length], 'loan_amnt': [loan_amnt], 'loan_int_rate': [loan_int_rate], 'loan_percent_income': [loan_percent_income]
            , 'cb_person_default_on_file': [cb_person_default_on_file], 'cb_person_cred_hist_length': [cb_person_cred_hist_length], 'person_home_ownership': [person_home_ownership]
            , 'loan_intent': [loan_intent], 'loan_grade': [loan_grade]}
    df = pd.DataFrame(data)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df = sc.fit_transform(df)

    
    import pickle
    
    
    pipe = pickle.load(open('pipe.pkl','rb'))

    if btn:
        result = pipe.predict(df)
        if result[0] == 0:
            st.success("Loan REJECTED")
        else: 
            st.success("Loan APPROVED")

    