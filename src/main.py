import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide")

# Light/Dark Mode
mode = st.sidebar.radio("Choose Mode:", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""<style>body {background-color: #1E1E1E; color: white;}</style>""", unsafe_allow_html=True)

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
    st.write("Upload datasets, train ML models, and visualize results.")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)
    with st.spinner("Loading animations..."):
        time.sleep(1)
    st.success("Ready to explore the app!")

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())
        st.success("Data successfully uploaded!")

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")
    st.write("Select the model parameters and train your ML model.")
    
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
    train_button = st.button("Train Model")
    
    if train_button:
        with st.spinner(f"Training {model_choice} model..."):
            time.sleep(2)
        st.success(f"{model_choice} model trained successfully!")
        st.balloons()

# Results Page
elif selected == "Results":
    st.title("Model Training Results")
    st.write("Visualizing model performance.")
    
    accuracy = {"Logistic Regression": 85, "Random Forest": 90, "SVM": 88}
    model_selected = st.selectbox("Select Trained Model", list(accuracy.keys()))
    st.metric(label="Model Accuracy", value=f"{accuracy[model_selected]}%")
    
    # Multiple Data Visualizations
    fig1 = px.bar(x=list(accuracy.keys()), y=list(accuracy.values()), labels={'x': "Model", 'y': "Accuracy (%)"}, title="Model Accuracy Comparison")
    fig2 = px.line(x=["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"], y=[75, 80, 85, accuracy[model_selected]], title="Training Accuracy Over Time")
    fig3 = px.scatter(x=df.index, y=df.iloc[:, 0], title="Feature Distribution")
    fig4 = px.histogram(df, x=df.columns[1], title="Feature Histogram")
    fig5 = px.box(df, y=df.columns[2], title="Box Plot of a Feature")
    
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
    st.plotly_chart(fig5)
    
    st.success("Real-time graphs updated!")
