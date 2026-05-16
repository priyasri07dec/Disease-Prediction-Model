import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
rf_model = joblib.load("disease_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")


#Load the training data to get the feature names
training_data = pd.read_csv("Training.csv")
training_data = training_data.dropna(axis =1, how ='all')  # Drop rows with missing values
features_names = training_data.drop("prognosis", axis=1).columns

# Streamlit app
st.title("Disease Prediction App")
st.write("Enter the symptoms to predict the disease:")

# multiselect for symptoms
selected_symptoms = st.multiselect("Select Symptoms", features_names)

# Create a DataFrame for the input symptoms
input_data = pd.DataFrame(columns=features_names)
for symptom in features_names:
    input_data[symptom] = [1 if symptom in selected_symptoms else 0]    
# Predict button
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Make prediction
        prediction = rf_model.predict(input_data)
        predicted_disease = encoder.inverse_transform(prediction)[0]
        st.success(f"The predicted disease is: {predicted_disease}")

