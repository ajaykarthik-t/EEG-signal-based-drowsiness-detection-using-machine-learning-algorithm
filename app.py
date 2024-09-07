
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import scipy.io as sio

# Load the pre-trained model and scaler
model_path = 'random_forest_model.pkl'
scaler_path = 'scaler.pkl'

try:
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please train the model first.")
    st.stop()

st.title('EEG Signal-based Drowsiness Detection')

# File uploader for new EEG data
uploaded_file = st.file_uploader("Upload your EEG data file (.mat)", type="mat")

if uploaded_file:
    st.write("File uploaded successfully.")

    # Load and preprocess the data
    mat_data = sio.loadmat(uploaded_file)
    eeg_data = mat_data['EEGsample']
    subindex = mat_data['subindex']
    substate = mat_data['substate']
    
    # Convert substate to a list of labels
    labels = [f"Sample {i+1}" for i in range(len(subindex))]

    # Dropdown to select the sample to predict
    selected_sample = st.selectbox("Select the sample to predict:", labels)

    # Get the index of the selected sample
    sample_index = labels.index(selected_sample)
    
    # Extract the selected sample data
    sample_data = eeg_data[sample_index, :]

    if sample_data.size > 0:
        # Standardize the selected sample
        new_features_scaled = scaler.transform(sample_data.reshape(1, -1))
        
        # Predict class probabilities
        probabilities = clf.predict_proba(new_features_scaled)
        prediction = clf.predict(new_features_scaled)
        
        st.write(f"Predicted class: {'Drowsy' if prediction[0] == 1 else 'Active'}")

        # Display the prediction probabilities
        st.write("Prediction probabilities:")
        st.write(f"Drowsy: {probabilities[0][1]:.2f}")
        st.write(f"Active: {probabilities[0][0]:.2f}")
        
        # Plot the selected sample data
        st.write("EEG Sample Data Visualization:")
        plt.figure(figsize=(10, 4))
        plt.plot(sample_data)
        plt.title('EEG Sample Data')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        st.pyplot(plt)
    else:
        st.write("Selected sample data is empty or invalid.")

