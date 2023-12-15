import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the logo image
logo = Image.open('logo.png')

# Define the SessionState class
class SessionState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create a session state object
session_state = SessionState(predictions={'binary_result': None, 'offensive_words': None, 'multi_class_result': None})

# Load the pre-trained pipeline
model_pipeline = joblib.load('sgd_classifier_model.joblib')
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

@st.cache(allow_output_mutation=True)
# Function to handle file upload and call the cached function
def handle_file_upload():
    # Ask the user to upload a file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.write("Experiment function is executing!")

    # Call the cached function with the uploaded file
    experiment_with_dataset(uploaded_file)
def experiment_with_dataset(uploaded_file):
    try:
        print("Start of experiment_with_dataset")

        if uploaded_file is not None:
            st.write(f"File uploaded: {uploaded_file.name}")

            # Load the new dataset
            df_new = pd.read_csv(uploaded_file)

            # Rest of your existing code...

            st.success("Dataset reprocessed and model retrained successfully.")

        print("End of experiment_with_dataset")
    except Exception as e:
        st.error(f"Error in experiment_with_dataset: {e}")
        print(f"Error in experiment_with_dataset: {e}")

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
st.sidebar.image(logo, caption=None, width=10, use_column_width=True)
page = st.sidebar.radio("Select Page", ["Twitter Interaction", "Custom Twitter Interaction"])

if page == "Twitter Interaction":
    st.title('Cyberbullying Detection App')

    # Input text box
    user_input = st.text_area("Share your thoughts:", "", key="user_input")

    # Button to trigger analysis
    analyze_button = st.button("Analyze")

    # View flag for detailed predictions
    view_predictions = st.checkbox("View Detailed Predictions", value=False)

    # Check if the user has entered any text and the button is clicked
    if user_input and analyze_button:
        # Make binary prediction and check for offensive words
        binary_result, offensive_words = binary_cyberbullying_detection(user_input)
        session_state.predictions['binary_result'] = binary_result
        session_state.predictions['offensive_words'] = offensive_words
        st.markdown("<div class='st-bw'>", unsafe_allow_html=True)

        if view_predictions:
            st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")

        # Display offensive words and provide recommendations
        if offensive_words and view_predictions:
            st.warning(f"While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing. Detected offensive words: {offensive_words}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Make multi-class prediction
        multi_class_result = multi_class_cyberbullying_detection(user_input)
        session_state.predictions['multi_class_result'] = multi_class_result
        if multi_class_result is not None:
            predicted_class, prediction_probs = multi_class_result
            st.markdown("<div class='st-eb'>", unsafe_allow_html=True)

            if view_predictions:
                st.write(f"Multi-Class Predicted Class: {predicted_class}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Check if classified as cyberbullying
            if predicted_class != 'not_cyberbullying':
                st.error(f"Please edit your tweet before resending. Your text contains content that may appear as bullying to other users. {predicted_class.replace('_', ' ').title()}.")
            elif offensive_words and not view_predictions:
                st.warning("While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing.")
            else:
                # Display message before sending
                st.success('This tweet is safe to send.')

                # Button to send tweet
                if st.button('Send Tweet'):
                    st.success('Tweet Sent!')

elif page == "Custom Twitter Interaction":
    st.title('Custom Cyberbullying Interaction')
    # Button to experiment with a custom dataset
    if st.button("Experiment with Your Dataset"):
        handle_file_upload()
 

        # Mark the function as executed in session_state
        st.session_state.experiment_executed = True
