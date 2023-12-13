
import streamlit as st
import re
import joblib
import numpy as np
import nltk
nltk.download('wordnet')
# Download NLTK resources
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the SGD classifier and TF-IDF vectorizer for binary cyberbullying detection
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
tfidf_vectorizer_binary = joblib.load('tfidf_vectorizer.joblib')

# Load the models for multi-class cyberbullying detection
models = {
    'Random Forest': joblib.load('random_forest_model.joblib'),
    'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
    'Stochastic Gradient Descent': joblib.load('sgd_model.joblib'),
    'Naive Bayes': joblib.load('naive_bayes_model.joblib'),
    'Support Vector Machine': joblib.load('svm_model.joblib'),
}

# Load the TF-IDF vectorizer for multi-class cyberbullying detection
tfidf_vectorizer_multiclass = joblib.load('tfidf_vectorizer_multiclass.joblib')

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

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = tfidf_vectorizer_binary.transform([preprocessed_text])

        # Make prediction
        prediction = sgd_classifier.predict(text_tfidf)

        return prediction[0]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = tfidf_vectorizer_multiclass.transform([preprocessed_text])

        # Make predictions on the sample text for each model
        predictions = {}
        for model_name, model in models.items():
            prediction_encoded = model.predict(text_tfidf)
            prediction = label_encoder.inverse_transform(prediction_encoded)
            predictions[model_name] = prediction[0]

        return predictions
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Enter a text:", "")

# Check if the user has entered any text
if user_input:
    # Make prediction for binary cyberbullying detection
    binary_prediction = binary_cyberbullying_detection(user_input)

    # Display the binary prediction
    st.write(f"Binary Prediction: {'Cyberbullying' if binary_prediction == 1 else 'Not Cyberbullying'}")

    # If cyberbullying is detected, make prediction for multi-class cyberbullying detection
    if binary_prediction == 1:
        # Make prediction for multi-class cyberbullying detection
        multi_class_prediction = multi_class_cyberbullying_detection(user_input)
        
        # Display the specific category prediction
        st.write(f"Specific Category: {multi_class_prediction}")
