import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide")

# Apply custom CSS for dark mode toggle and hover effects
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff5722 !important;
            color: white !important;
        }
        .stDataFrame {border-radius: 10px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation with Dark Mode Toggle
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🌗 Mode</h1>", unsafe_allow_html=True)
    dark_mode = st.checkbox("Enable Dark Mode", value=True)
    
    selected = option_menu(
        "Main Menu",
        ["Home", "Upload Data", "Train Model", "Results"],
        icons=["house", "cloud-upload", "cpu", "bar-chart-line"],
        menu_icon="app-indicator",
        default_index=0,
    )

if dark_mode:
    st.markdown(
        """
        <style>
            body {
                background-color: #121212;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Data persistence
if "df" not in st.session_state:
    st.session_state.df = None

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
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Store in session state
        st.write("### Data Preview:")
        st.dataframe(df.head())

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
    else:
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
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
    else:
        df = st.session_state.df
        st.write("Here you can visualize model performance.")
        accuracy = {"Logistic Regression": 85, "Random Forest": 90, "SVM": 88}
        model_selected = st.selectbox("Select Trained Model", list(accuracy.keys()))
        st.metric(label="Model Accuracy", value=f"{accuracy[model_selected]}%")
        
        # Real-time graphs
        fig1 = px.histogram(df, x=df.columns[0], title="Data Distribution")
        fig2 = px.line(df, x=df.index, y=df.iloc[:, 1], title="Feature Trend")
        fig3 = px.scatter(x=df.index, y=df.iloc[:, 0], title="Feature Distribution")
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
