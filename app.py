import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
# Download NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# Load the entire pipeline (including TfidfVectorizer and SGDClassifier)
model_pipeline = joblib.load('sgd_classifier_model.joblib')

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for binary cyberbullying detection
def binary_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict([preprocessed_text])

        return prediction[0]
    except Exception as e:
        st.error(f"Error in binary_cyberbullying_detection: {e}")
        return None

# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction
        decision_function_values = sgd_classifier.decision_function([preprocessed_text])[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(decision_function_values)

        # Get the predicted class label using the label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_class_label, decision_function_values
    except Exception as e:
        st.error(f"Error in multi_class_cyberbullying_detection: {e}")
        return None

# Streamlit UI
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Enter a tweet:", "")

# Check if the user has entered any text
if user_input:
    # Make binary prediction
    binary_result = binary_cyberbullying_detection(user_input)
    st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")

    # Make multi-class prediction
    multi_class_result = multi_class_cyberbullying_detection(user_input)
    if multi_class_result is not None:
        predicted_class, prediction_probs = multi_class_result
        st.write(f"Multi-Class Predicted Class: {predicted_class}")
        st.write(f"Decision Function Values: {prediction_probs}")

        # Check if classified as cyberbullying
        if predicted_class != 'not_cyberbullying':
            st.error("Cyberbullying detected! Please edit your tweet before resending.")
        else:
            # Button to send tweet
            if st.button('Send Tweet'):
                st.success('Tweet Sent!')
