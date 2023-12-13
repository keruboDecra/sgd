
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

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import joblib
import re
import numpy as np
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import joblib
import re
import numpy as np
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the SGD classifier and TF-IDF vectorizer
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for multi-class and binary cyberbullying detection
def detect_cyberbullying(text, binary_threshold=0.5):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction on multi-class
        decision_function_values = sgd_classifier.decision_function([preprocessed_text])
        predicted_class_index = np.argmax(decision_function_values)
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        # Make prediction on binary
        binary_prediction = 1 if np.max(decision_function_values) > binary_threshold else 0

        return predicted_class_label, decision_function_values, binary_prediction
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
    result = detect_cyberbullying(user_input)

    # Display the prediction
    if result is not None:
        predicted_class, decision_function_values, binary_prediction = result
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Decision Function Values: {decision_function_values}")
        st.write(f"Binary Prediction: {'Cyberbullying' if binary_prediction == 1 else 'Not Cyberbullying'}")
