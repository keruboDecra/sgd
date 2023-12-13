import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image

# Download NLTK resources
nltk.download('wordnet')
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

# Set page title and icon
st.set_page_config(
    page_title="Cyberbullying Detection App",
    page_icon="🕵️",
)

# Apply styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            color: #333333;
        }
        .st-bw {
            background-color: #ffffff;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .st-bw:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .st-eb {
            background-color: #e6f7ff;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .st-eb:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .st-error {
            background-color: #ffcccc;
            color: #990000;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .st-success {
            background-color: #ccffcc;
            color: #006600;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .logo {
            max-width: 40%;
            margin-top: 20px;
        }
        h1 {
            color: #0077cc;
        }
        .stTextInput textarea {
            color: #333333 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.image(logo, caption=None, width=40, use_column_width=True)
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Compose your tweet:", "", key="user_input")

# Check if the user has entered any text
if user_input:
    # Make binary prediction and check for offensive words
    binary_result, offensive_words = binary_cyberbullying_detection(user_input)
    st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
    st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")

    # Display offensive words and provide recommendations
    if offensive_words:
        st.warning("Offensive words detected! Please consider editing the following words:")
        st.write(offensive_words)

    st.markdown("</div>", unsafe_allow_html=True)

    # Make multi-class prediction
    multi_class_result = multi_class_cyberbullying_detection(user_input)
    if multi_class_result is not None:
        predicted_class, prediction_probs = multi_class_result
        st.markdown("<div class='st-eb'>", unsafe_allow_html=True)
        st.write(f"Multi-Class Predicted Class: {predicted_class}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Check if classified as cyberbullying
        if predicted_class != 'not_cyberbullying':
            st.error("Cyberbullying detected! Please edit your tweet before resending.")
        else:
            # Button to send tweet
            if st.button('Send Tweet'):
                st.success('Tweet Sent!')

