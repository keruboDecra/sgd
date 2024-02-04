
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
from tweepy import Stream
from tweepy import OAuthHandler


# Twitter API credentials
api_key = "HBDNk66DlCsiPLmXWv43U7xBE"
api_key_secret = "BNejVusnkg9GCwfgVMTRmyN5CKADO62v969cUKhPzMatlPrn5B"
access_token = "1729269231722594305-iB1xl33Ou4GmrlimOOdJ0vmx2FD8a4"
access_token_secret = "bE0b5A0gMsujnsIhMn6QEGqiVa6K6rkZLLeJXxWgDfUEo"

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model_pipeline = joblib.load('sgd_classifier_model.joblib')
new_model_pipeline = None

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the logo image
logo = Image.open('logo.png')

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

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        st.error(f"Error in binary_cyberbullying_detection: {e}")
        return None, None

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




def classify_highlighted_text():
    st.title('Cyberbullying Detection App')

    # Receive selected text from Chrome extension
    selected_text = st.session_state.selected_text
    print(f"Received selected text: {selected_text}")  # Add this line for debugging

    if selected_text:
        st.write(f"Selected Text: {selected_text}")

        # Perform classification using your existing functions
        binary_result, offensive_words = binary_cyberbullying_detection(selected_text)
        multi_class_result = multi_class_cyberbullying_detection(selected_text)

        # Display classification results in Streamlit
        st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")
        st.write(f"Multi-Class Predicted Class: {multi_class_result[0]}")

# Check if the app is being used by the Chrome extension
if 'selected_text' in st.session_state:
    classify_highlighted_text()



