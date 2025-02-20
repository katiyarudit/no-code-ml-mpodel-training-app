import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide")

# Dark Mode Toggle
st.markdown(
    """
    <style>
        body { background-color: #121212; color: white; }
        .stButton>button:hover { background-color: #ff6b6b; color: white; }
        .stSelectbox:hover, .stFileUploader:hover { border-color: #ff6b6b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Upload Data", "Train Model", "Results"],
        icons=["house", "cloud-upload", "cpu", "bar-chart-line"],
        menu_icon="app-indicator",
        default_index=0,
    )

# Home Page with Animation
if selected == "Home":
    st.title("🚀 Welcome to ML Model Training App 🤖")
    st.write("This app allows you to upload datasets, train machine learning models, and visualize results.")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)
    with st.spinner("Loading Cool Features..."):
        time.sleep(2)
    st.success("Everything is Ready! Start Exploring 🚀")

# Session State for Data Persistence
if "data" not in st.session_state:
    st.session_state["data"] = None

# Upload Data Page
elif selected == "Upload Data":
    st.title("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df  # Store in session state
        st.write("### Data Preview:")
        st.dataframe(df.head())

# Train Model Page
elif selected == "Train Model":
    st.title("⚙️ Train Your Model")
    st.write("Select the model parameters and train your ML model.")
    
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
    train_button = st.button("Train Model")
    
    if train_button and st.session_state["data"] is not None:
        st.success(f"{model_choice} model is being trained...")
        st.progress(100)
        st.balloons()
    elif train_button:
        st.warning("Please upload a dataset first!")

# Results Page with Real-Time Graphs
elif selected == "Results":
    st.title("📊 Model Training Results")
    st.write("Here you can visualize model performance with real-time charts.")
    
    if st.session_state["data"] is not None:
        df = st.session_state["data"]
        st.metric(label="Rows in Dataset", value=df.shape[0])
        st.metric(label="Columns in Dataset", value=df.shape[1])
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x=df.columns[0], title="Feature Distribution")
            st.plotly_chart(fig1)
        with col2:
            fig2 = px.box(df, y=df.columns[1], title="Box Plot Analysis")
            st.plotly_chart(fig2)
    else:
        st.warning("No data available! Please upload a dataset first.")
