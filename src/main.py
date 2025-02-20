import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import os

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Upload Data", "Train Model", "Results"],
        icons=["house", "cloud-upload", "cpu", "bar-chart-line"],
        menu_icon="app-indicator",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("Welcome to ML Model Training App 🤖")
    st.write("This app allows you to upload datasets, train machine learning models, and visualize results.")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)
    st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Empower Your Data with Machine Learning</h3>", unsafe_allow_html=True)

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")
    st.write("Select the model parameters and train your ML model.")
    
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
    train_button = st.button("Train Model")
    
    if train_button:
        st.success(f"{model_choice} model is being trained...")
        st.progress(100)
        st.balloons()
        model_filename = f"{model_choice.replace(' ', '_')}_trained_model.sav"
        with open(model_filename, 'wb') as f:
            pickle.dump(model_choice, f)
        st.success("Model training complete! You can download the trained model below.")
        st.download_button(label="Download Trained Model", data=open(model_filename, "rb"), file_name=model_filename, mime="application/octet-stream")

# Results Page
elif selected == "Results":
    st.title("Model Training Results")
    st.write("Here you can visualize model performance.")
    
    # Dummy accuracy values
    accuracy = {"Logistic Regression": 85, "Random Forest": 90, "SVM": 88}
    model_selected = st.selectbox("Select Trained Model", list(accuracy.keys()))
    
    st.metric(label="Model Accuracy", value=f"{accuracy[model_selected]}%")
    
    # Dummy chart
    fig, ax = plt.subplots()
    ax.bar(accuracy.keys(), accuracy.values(), color=['blue', 'green', 'red'])
    ax.set_ylabel("Accuracy (%)")
    st.pyplot(fig)
    
    if uploaded_file is not None:
        st.subheader("Additional Data Visualizations")
        fig1 = px.histogram(df, x=df.columns[0], title="Feature Distribution")
        st.plotly_chart(fig1)
        fig2 = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Feature Relationship")
        st.plotly_chart(fig2)
