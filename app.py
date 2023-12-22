import streamlit as st
import numpy as np
import pickle

# Load the model
lr = pickle.load(open('./assets/newLR.pkl', 'rb'))
df = pickle.load(open('./assets/ad.pkl', 'rb'))

st.title('Sales Prediction App')

# User input
tv = st.slider('Money Spend on TV Advertising', min_value=0, max_value=300, step=1)
news = st.slider('Money Spend on NEWSPAPER Advertising', min_value=0, max_value=115, step=1)
radio = st.slider('Money Spend on RADIO Advertising', min_value=0, max_value=50, step=1)


if st.button('Predict Sales'):
    # Ensure 'tv' is a numeric value
    query = np.array([tv,news,radio])  
    query = query.reshape(1,3)
    predicted_sales = lr.predict(query)

    st.title("The Sales Growth is " + str(int(predicted_sales[0])) + "%")  # Use [0] to get the scalar value
