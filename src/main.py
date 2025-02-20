import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
import pickle
from io import BytesIO

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
    st.markdown("### Train machine learning models with ease!")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)
    st.markdown("---")
    st.markdown("#### Why Use This App?")
    st.write("✅ Easy dataset upload")
    st.write("✅ Train ML models quickly")
    st.write("✅ Visualize results dynamically")
    st.write("✅ Download trained models for reuse")
    st.balloons()

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.write("### Data Preview:")
        st.dataframe(df.head())

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")
    if 'data' in st.session_state:
        df = st.session_state['data']
        model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
        train_button = st.button("Train Model")
        
        if train_button:
            with st.spinner("Training model..."):
                time.sleep(2)
                model = {"Logistic Regression": "LR Model", "Random Forest": "RF Model", "SVM": "SVM Model"}[model_choice]
                st.session_state['trained_model'] = model
                model_file = BytesIO()
                pickle.dump(model, model_file)
                st.session_state['model_file'] = model_file
                st.success(f"{model_choice} model trained successfully!")
                st.balloons()
    else:
        st.warning("Please upload a dataset first.")

# Results Page
elif selected == "Results":
    st.title("Model Training Results")
    if 'data' in st.session_state and 'trained_model' in st.session_state:
        df = st.session_state['data']
        st.write("### Data Visualization")
        
        fig1 = px.histogram(df, x=df.columns[0], title="Feature Distribution")
        fig2 = px.box(df, y=df.columns[0], title="Box Plot")
        fig3 = px.scatter(df, x=df.index, y=df.iloc[:, 0], title="Feature Scatter Plot")
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        st.write("### Download Trained Model")
        if 'model_file' in st.session_state:
            st.download_button("Download Model", st.session_state['model_file'].getvalue(), "trained_model.pkl", "application/octet-stream")
    else:
        st.warning("Please train a model first.")
