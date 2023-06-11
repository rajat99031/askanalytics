import pandas as pd
import streamlit as st
import datetime
import pickle
import numpy as np
import joblib

titanic = pd.read_csv(r"C:\TEST\PJ\titanic.csv")

encode_dict = {
    "Sex_female": {'Male': 0, 'Female': 1},
    "age_encoded": {'0_10': 1, '10_30': 2, '30_50': 3,'50_100' : 4}
}

st.write(
    """
     ### SURVIVED Prediction
    """
)
st.dataframe(titanic.head())

def model_pred(PClass,Sex_female, age_encoded, Fare_M2):
    with open("C:\TEST\PJ\Logistic_model.pkl", 'rb') as file:
        reg_model = joblib.load(file)

        Sex_female = encode_dict['Sex_female'][Sex_female]
        age_encoded = encode_dict['age_encoded'][age_encoded]
        input_features = [[PClass,Sex_female, age_encoded, Fare_M2]]
        
        # input_features = np.array(input_features)

        return reg_model.predict(input_features)



col1, col2 = st.columns(2)

PClass = col1.slider("select the PClass ",
                     1, 3, step=1)

age_encoded = col2.selectbox("Select the age_encoded",
                           ["0_10", "10_30","30_50","50_100"])

Sex_female = col2.selectbox("Select the sex",
                           ["Male", "Female"])

Fare_M2 = col1.slider("select the Fare_M2 ",
                     0.00, 1.00, step=0.01)


if (st.button("Survived or Unsurvived")):
    
    survived = model_pred(PClass,Sex_female, age_encoded, Fare_M2)

    st.text(str(survived))
    if (survived==0) :
        st.text("Passenger did not survive ")
    else :
        st.text("Passenger survived ") 
