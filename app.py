import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
            background: linear-gradient(90deg, #5A5A5A 0%, #333333 100%);
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .st-bw {
            background-color: #666666;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .st-bw:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .st-eb {
            background-color: #4C4C4C;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .st-eb:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .st-error {
            background-color: #FF6666;
            color: #990000;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .st-success {
            background-color: #66FF66;
            color: #006600;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .logo {
            max-width: 30%;
            margin-top: 20px;
        }
        h1 {
            color: #CC99FF;
        }
        .stTextInput textarea {
            color: #333333 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.image(logo, caption=None, width=10, use_column_width=True)
st.title('Cyberbullying Detection App')

# Experimentation page
if st.button("Experiment"):
    st.header("Experimentation Page")
    st.write("Upload your CSV file for training and prediction.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        df_exp = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.write("First few rows of the uploaded dataset:")
        st.write(df_exp.head())

        # Create a new DataFrame for preprocessed data
        df_preprocessed_exp = df_exp.copy()

        # Apply text preprocessing to the 'tweet_text' column
        df_preprocessed_exp['cleaned_text'] = df_preprocessed_exp['tweet_text'].apply(preprocess_text)

        # Display the cleaned and preprocessed data
        st.write("Cleaned and Preprocessed Data:")
        st.write(df_preprocessed_exp[['tweet_text', 'cleaned_text']].head())

        # Encode the target variable
        df_preprocessed_exp['encoded_label'] = label_encoder.transform(df_preprocessed_exp['cyberbullying_type'])

        # Display the encoded labels
        st.write("Encoded Labels:")
        st.write(df_preprocessed_exp[['cyberbullying_type', 'encoded_label']].head())

        # Split the data into training and testing sets
        X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
            df_preprocessed_exp['cleaned_text'],
            df_preprocessed_exp['encoded_label'],
            test_size=0.2,
            random_state=42
        )

        # Display the shapes of the training and testing sets
        st.write("Shapes of Training and Testing Sets:")
        st.write(f"X_train shape: {X_train_exp.shape}, y_train shape: {y_train_exp.shape}")
        st.write(f"X_test shape: {X_test_exp.shape}, y_test shape: {y_test_exp.shape}")

        # Train and evaluate each model on the user's dataset
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train_exp, y_train_exp)

            # Make predictions on the test data
            y_pred_exp = model.predict(X_test_exp)

            # Evaluate the model
            accuracy_exp = accuracy_score(y_test_exp, y_pred_exp)
            st.write(f"\n{model_name} Accuracy on User's Dataset: {accuracy_exp:.2f}")

            # Display additional evaluation metrics
            st.write("Classification Report:")
            st.write(classification_report(y_test_exp, y_pred_exp))
