import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="ML Model Training App", page_icon="🤖", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = st.radio("Navigation", ["Home", "Upload Data", "Train Model", "Results"])

# Home Page
if selected == "Home":
    st.title("Welcome to ML Model Training App 🤖")
    st.write("This app allows you to upload datasets, train machine learning models, and visualize results.")

# Upload Data Page
elif selected == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df  # Store DataFrame in session state
        st.write("### Data Preview:")
        st.dataframe(df)

# Train Model Page
elif selected == "Train Model":
    st.title("Train Your Model")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first!")
    else:
        df = st.session_state["df"]
        model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
        train_button = st.button("Train Model")
        
        if train_button:
            st.success(f"{model_choice} model is being trained...")
            st.progress(100)
            st.balloons()
            
            model = {"Logistic Regression": "LR Model", "Random Forest": "RF Model", "SVM": "SVM Model"}
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model[model_choice], f)
            st.download_button("Download Trained Model", data=open("trained_model.pkl", "rb"), file_name="trained_model.pkl")

# Results Page
elif selected == "Results":
    st.title("Model Training Results")

    if "df" not in st.session_state:
        st.warning("No dataset found! Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.write("## Data Visualizations")
        
        fig1 = px.histogram(df, x=df.columns[0], title="Feature Distribution")  
        st.plotly_chart(fig1)
        
        if len(df.columns) > 1:
            fig2 = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Feature Correlation")
            st.plotly_chart(fig2)
            
            fig3 = px.line(df, x=df.index, y=df.iloc[:, 0], title="Trend Line")
            st.plotly_chart(fig3)
            
            fig4 = px.box(df, y=df.iloc[:, 0], title="Box Plot of First Feature")
            st.plotly_chart(fig4)
