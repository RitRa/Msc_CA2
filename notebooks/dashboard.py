import streamlit as st


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title("This is a test")
    st.text("this some textx")
   
with dataset:
    st.header("Fertiliser dataset")
    
with features:
    st.header("features")
    
with model_training:
    st.header("models")
    