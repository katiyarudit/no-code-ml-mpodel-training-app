import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt

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

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # Try reading normally
        except:
            uploaded_file.seek(0)  # Reset pointer
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")  # Try different encoding
        
        if df.empty:
            st.warning("⚠️ The uploaded CSV file is empty or not read correctly.")
        else:
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
