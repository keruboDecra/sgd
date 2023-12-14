import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

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

# Load the default model
default_model = joblib.load('sgd_classifier_model.joblib')

# Streamlit UI
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

# Option to upload a CSV file
uploaded_file = st.file_uploader("Upload CSV File for Experimentation", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded dataset
    df_uploaded = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded dataset
    st.subheader("First few rows of the uploaded dataset:")
    st.write(df_uploaded.head())

    # Preprocess the uploaded data
    df_uploaded['cleaned_text'] = df_uploaded['tweet_text'].apply(preprocess_text)

    # Encode the target variable
    df_uploaded['encoded_label'] = label_encoder.transform(df_uploaded['cyberbullying_type'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_uploaded['cleaned_text'],
        df_uploaded['encoded_label'],
        test_size=0.2,
        random_state=42
    )

    # Display the shapes of the training and testing sets
    st.subheader("Shapes of Training and Testing Sets for the Uploaded Dataset:")
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Train the model on the uploaded dataset
    model_pipeline_uploaded = make_pipeline(TfidfVectorizer(), SGDClassifier(random_state=42))
    model_pipeline_uploaded.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_uploaded = model_pipeline_uploaded.predict(X_test)

    # Evaluate the model
    accuracy_uploaded = accuracy_score(y_test, y_pred_uploaded)
    st.subheader(f"Accuracy on the Uploaded Dataset: {accuracy_uploaded:.2f}")

    # Display additional evaluation metrics
    st.subheader("Classification Report on the Uploaded Dataset:")
    st.write(classification_report(y_test, y_pred_uploaded))

    # Option to save the trained model
    save_model = st.checkbox("Save the Trained Model")
    if save_model:
        # Save the model pipeline to a file
        joblib.dump(model_pipeline_uploaded, 'uploaded_model.joblib', protocol=4)
        st.success("Trained model saved as 'uploaded_model.joblib'")
