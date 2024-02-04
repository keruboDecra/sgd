import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from io import BytesIO
from PIL import Image
import sys  # Import the sys module
# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.base import clone
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the logo image
logo = Image.open('logo.png')

# Function to clean and preprocess text
def preprocess_text(selected_text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', selected_text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for binary cyberbullying detection
def binary_cyberbullying_detection(selected_text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(selected_text)

        # Make prediction using the loaded pipeline
        prediction = sgd_classifier.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        st.error(f"Error in binary_cyberbullying_detection: {e}")
        return None, None

# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(selected_text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(selected_text)

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
user_input = st.text_area("Share your thoughts:", "", key="user_input")

# Check if selected text is provided from Chrome extension
selected_text = st.session_state.get('selected_text')

if selected_text:
    st.write(f"Selected Text: {selected_text}")

    # Perform classification using your existing functions
    binary_result, offensive_words = binary_cyberbullying_detection(selected_text)
    multi_class_result = multi_class_cyberbullying_detection(selected_text)

    # Display classification results in Streamlit
    st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")
    st.write(f"Multi-Class Predicted Class: {multi_class_result[0]}")

# Perform classification based on user input
if user_input:
    binary_result, offensive_words = binary_cyberbullying_detection(user_input)
    multi_class_result = multi_class_cyberbullying_detection(user_input)

    # Display classification results in Streamlit
    st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")
    st.write(f"Multi-Class Predicted Class: {multi_class_result[0]}")
