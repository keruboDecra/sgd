
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


model = joblib.load('sgd_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')


def binary_cyberbullying_detection(text, model, vectorizer):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(text_tfidf)

        return prediction[0]
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage:
user_input = "This is a tweet containing cyberbullying language."
prediction = binary_cyberbullying_detection(user_input, model, vectorizer)

if prediction is not None:
    print(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")
