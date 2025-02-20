import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark mode and hover effects
css = """
<style>
    body { background-color: #0d1117; color: white; }
    .stSidebar { background-color: #161b22; }
    .stButton > button { background-color: #1f6feb; color: white; border-radius: 8px; }
    .stButton > button:hover { background-color: #3a9dfb; }
    .stSelectbox > div:hover { background-color: #21262d; }
    .stFileUploader > div:hover { background-color: #21262d; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Upload Data", "Train Model", "Results"],
        icons=["house", "cloud-upload", "cpu", "bar-chart-line"],
        menu_icon="app-indicator",
        default_index=0,
    )

# Session state for file persistence
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

# Home Page
if selected == "Home":
    st.title("Welcome to ML Model Training App 🤖")
    st.write("This app allows you to upload datasets, train machine learning models, and visualize results.")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.uploaded_data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(st.session_state.uploaded_data.head())

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

# Results Page
elif selected == "Results":
    st.title("Model Training Results")
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
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
