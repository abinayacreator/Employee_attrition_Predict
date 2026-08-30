import streamlit as st
import pandas as pd, joblib
st.set_page_config(page_title='Employee Attrition Prediction',page_icon='👥')
st.title('Employee Attrition Prediction')
st.write('Enter employee details and predict the probability of attrition.')
model=joblib.load('models/employee_attrition_model.pkl')
df=pd.read_csv('data/employee_attrition.csv')
features=[c for c in df.columns if c not in ['Attrition','EmployeeCount','EmployeeNumber','Over18','StandardHours']]
with st.form('prediction'):
    values={}
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            values[c]=st.number_input(c,value=float(df[c].median()))
        else:
            values[c]=st.selectbox(c,sorted(df[c].dropna().unique()))
    submitted=st.form_submit_button('Predict Attrition')
if submitted:
    x=pd.DataFrame([values])
    prob=model.predict_proba(x)[0,1]
    pred=model.predict(x)[0]
    st.subheader('Prediction Result')
    st.write('Likely to Leave' if pred==1 else 'Likely to Stay')
    st.metric('Attrition Probability',f'{prob:.2%}')
