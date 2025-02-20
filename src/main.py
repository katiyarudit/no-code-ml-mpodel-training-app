import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")


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
if selected == "Results":
    st.title("Model Training Results")

    uploaded_file = st.file_uploader("Upload a CSV file to visualize results", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())

        # Scatter Plot
        fig1 = px.scatter(df, x=df.index, y=df.columns[0], title="Feature Distribution")
        st.plotly_chart(fig1)

        # Line Chart
        fig2 = px.line(df, x=df.index, y=df.columns[1], title="Feature Trend")
        st.plotly_chart(fig2)

        # Histogram
        fig3 = px.histogram(df, x=df.columns[0], title="Feature Frequency")
        st.plotly_chart(fig3)
    else:
        st.warning("Upload a dataset to visualize graphs.")
#additional css for dark and light  mode
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    css = """
    <style>
    body { background-color: #121212; color: white; }
    .stSidebar { background-color: #1E1E1E; }
    </style>
    """
else:
    css = """
    <style>
    body { background-color: white; color: black; }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

