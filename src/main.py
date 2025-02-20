import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
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
    st.title("🚀 Welcome to the ML Model Training App")
    st.write("This app allows you to upload datasets, train machine learning models, and visualize results.")
    st.image("https://source.unsplash.com/800x400/?technology,machinelearning", use_column_width=True)

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.write("### Data Preview:")
        st.dataframe(df.head())

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")
    st.write("Select the model parameters and train your ML model.")
    
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
    train_button = st.button("Train Model")
    
    if train_button and "df" in st.session_state:
        model = {"model": model_choice, "data": st.session_state["df"].to_dict()}
        model_filename = "trained_model.pkl"
        
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        
        st.success(f"{model_choice} model has been trained successfully!")
        
        with open(model_filename, "rb") as file:
            st.download_button(
                label="Download Trained Model",
                data=file,
                file_name=model_filename,
                mime="application/octet-stream"
            )

# Results Page
elif selected == "Results":
    st.title("Model Training Results")
    st.write("Here you can visualize model performance.")
    
    if "df" in st.session_state:
        df = st.session_state["df"]
        fig1 = px.histogram(df, x=df.columns[0], title="Feature Distribution")
        fig2 = px.box(df, x=df.columns[0], title="Box Plot of Feature")
        fig3 = px.scatter(df, x=df.index, y=df.iloc[:, 0], title="Feature Scatter Plot")
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Please upload a dataset first.")
