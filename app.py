
import streamlit as st
import pickle
import numpy as np 
import pandas as pd


## load the instances that were created
with open('final_model.pkl','rb') as file:
    model = pickle.load(file)

with open('pca.pkl','rb') as file:
    pca = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

def prediction(input_data):
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    pred = model.predict(pca_data)[0]

    if pred==0:
        return 'Developing Country'
    elif pred==1:
        return 'Underdeveloped Country'
    else:
        return 'Developed Country'

def main():
    st.title('HELP International Foundation')
    st.subheader('This application will give the status of the country based on the socio-economic factors')
    child_mort = st.text_input('Enter the child mortality rate:')
    exports = st.text_input('Enter Exports (% GDP):')
    health = st.text_input('Enter expenditure on health (% GDP):')
    imports = st.text_input('Enter Imports (% GDP):')
    income = st.text_input('Enter average income:')
    inflation = st.text_input('Enter inflation:')
    life_expectancy = st.text_input('Enter life expectancy:')
    total_fertility = st.text_input('Enter fertility rate:')
    gdpp = st.text_input('Enter GDP per population:')

    input_list = [[child_mort , exports , health ,imports ,income ,inflation ,life_expectancy,total_fertility,gdpp]]
    
    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)
        
if __name__== '__main__':
        main()
