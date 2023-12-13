
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

# Load the SGD classifier and TF-IDF vectorizer
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for binary cyberbullying detection
def binary_cyberbullying_detection(text, vectorizer, classifier):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Transform the preprocessed text using the loaded vectorizer
        text_tfidf = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = classifier.predict(text_tfidf, model)

        return prediction[0]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Enter a text:", "")

# Check if the user has entered any text
if user_input:
    # Make prediction
    prediction = binary_cyberbullying_detection(user_input, tfidf_vectorizer, sgd_classifier)

    # Display the prediction
    if prediction is not None:
        st.write(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")
