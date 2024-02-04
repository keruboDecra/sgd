
import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
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


def experiment_with_dataset(uploaded_file):
    global new_model_pipeline  # Use the global variable

    print("Experiment function is executing!")

    try:
        if uploaded_file is not None:
            # Load the new dataset
            df_new = pd.read_csv(uploaded_file)
            st.write("Preview of the uploaded data:")
            st.write(df_new.head())

            st.write("Please wait as we train the model to your data...")

            # Create a new DataFrame for preprocessed data
            df_preprocessed_new = df_new.copy()

            # Apply text preprocessing to the 'tweet_text' column
            df_preprocessed_new['cleaned_text'] = df_preprocessed_new['tweet_text'].apply(preprocess_text)

            # Encode the target variable using the saved label encoder
            df_preprocessed_new['encoded_label'] = label_encoder.transform(df_preprocessed_new['cyberbullying_type'])

            # Split the new data into training and testing sets
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
                df_preprocessed_new['cleaned_text'],
                df_preprocessed_new['encoded_label'],
                test_size=0.2,
                random_state=42
            )

            # Create a new model pipeline
            new_model_pipeline = clone(model_pipeline)  # Copy the original model

            # Retrain the new model on the new training data
            new_model_pipeline.fit(X_train_new, y_train_new)

            # Save the updated pipeline to a new file path
            joblib.dump(new_model_pipeline, 'sgd_classifier_model_updated.joblib', protocol=4)

            # Optional: Print or return any relevant information
            st.success("Dataset reprocessed, and a new model trained and saved successfully.")
            return new_model_pipeline  # Return the trained model

    except Exception as e:
        st.error(f"Training failed, dataset structure incompatible. Check your dataset and try again. Error: {e}")
        sys.exit()
def new_binary_cyberbullying_detection(text):
    global new_model_pipeline  # Use the global variable

    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = new_model_pipeline.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        st.error(f"Error in new_binary_cyberbullying_detection: {e}")
        return None, None



def new_multi_class_cyberbullying_detection(text):
    global new_model_pipeline  # Use the global variable

    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction
        decision_function_values = new_model_pipeline.decision_function([preprocessed_text])[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(decision_function_values)

        # Get the predicted class label using the label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_class_label, decision_function_values
    except Exception as e:
        st.error(f"Error in new_multi_class_cyberbullying_detection: {e}")
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
            background: linear-gradient(green, #5A5A5A 0%, #333333 100%);
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .st-bw {
            background-color: sage;
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
        .stWarning {
            color: orange !important;
            background-color: transparent !important;
        }

        h1 {
            color: #Cc99FF;
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


def twitter_interaction_page():
    st.title('Cyberbullying Detection App')

        # Input text box
    user_input = st.text_area("Share your thoughts:", "", key="user_input")
    
    # Make binary prediction and check for offensive words
    binary_result, offensive_words = binary_cyberbullying_detection(user_input)

    # # View flag for detailed predictions
    # view_flagging_reasons = binary_result == 1
    # view_label = "View Flagging Reasons" if view_flagging_reasons else "Review Tweet Quality"
    # view_predictions = st.checkbox(view_label, value=False)

    view_flagging_reasons = binary_result == 1
    view_predictions = st.checkbox("View Flagging Reasons", value=view_flagging_reasons)
    

    
    # Check if the user has entered any text
    if user_input:
        st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
        
        # Display binary prediction only if "View Flagging Reasons" is checked
        if view_predictions and binary_result == 1:
            st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")
        
        # Check for offensive words and display warning
        if offensive_words and (view_predictions or binary_result == 0):
            # Adjust the warning message based on cyberbullying classification
            if binary_result == 1:
                st.warning(f"This tweet contains offensive language. Consider editing. Detected offensive words: {offensive_words}")
            else:
                st.warning(f"While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing. Detected offensive words: {offensive_words}")
    
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Make multi-class prediction
        multi_class_result = multi_class_cyberbullying_detection(user_input)
        if multi_class_result is not None:
            predicted_class, prediction_probs = multi_class_result
            st.markdown("<div class='st-eb'>", unsafe_allow_html=True)
            
            if view_predictions:
                st.write(f"Multi-Class Predicted Class: {predicted_class}")
    
            st.markdown("</div>", unsafe_allow_html=True)
    
            # Check if classified as cyberbullying
            if predicted_class != 'not_cyberbullying':
                st.error(f"Please edit your tweet before resending. Your text contains content that may appear as bullying to other users' {predicted_class.replace('_', ' ').title()}.")
            elif offensive_words and not view_predictions:
                st.warning("While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing.")
            else:
                # Display message before sending
                st.success('This tweet is safe to send.')
    
                # Button to send tweet
                if st.button('Send Tweet'):
                    st.success('Tweet Sent!')


def custom_twitter_interaction_page():
    st.title('Custom Cyberbullying Interaction')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Execute the experiment function and get the new model pipeline
        new_model_pipeline = experiment_with_dataset(uploaded_file)

        # Input text box
        user_input = st.text_area("Share your thoughts:", "", key="user_input")
        
        # Make binary prediction and check for offensive words
        binary_result, offensive_words = new_binary_cyberbullying_detection(user_input)


        view_flagging_reasons = binary_result == 1
        view_label = "View Flagging Reasons" if view_flagging_reasons else "Review Tweet Quality"
        view_predictions = st.checkbox(view_label, value=False)

        # Check if the user has entered any text
        if user_input:
            st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
            
            # Display binary prediction only if "View Flagging Reasons" is checked
            if view_predictions and binary_result == 1:
                st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")
            
            # Check for offensive words and display warning
            if offensive_words and (view_predictions or binary_result == 0):
                # Adjust the warning message based on cyberbullying classification
                if binary_result == 1:
                    st.warning(f"This tweet contains offensive language. Consider editing. Detected offensive words: {offensive_words}")
                else:
                    st.warning(f"While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing. Detected offensive words: {offensive_words}")
        
            st.markdown("</div>", unsafe_allow_html=True)
        
            # Make multi-class prediction
            multi_class_result = new_multi_class_cyberbullying_detection(user_input)
            if multi_class_result is not None:
                predicted_class, prediction_probs = multi_class_result
                st.markdown("<div class='st-eb'>", unsafe_allow_html=True)
                
                if view_predictions:
                    st.write(f"Multi-Class Predicted Class: {predicted_class}")
        
                st.markdown("</div>", unsafe_allow_html=True)
        
                # Check if classified as cyberbullying
                if predicted_class != 'not_cyberbullying':
                    st.error(f"Please edit your tweet before resending. Your text contains content that may appear as bullying to other users' {predicted_class.replace('_', ' ').title()}.")
                elif offensive_words and not view_predictions:
                    st.warning("While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing.")
                else:
                    # Display message before sending
                    st.success('This tweet is safe to send.')
        
                    # Button to send tweet
                    if st.button('Send Tweet'):
                        st.success('Tweet Sent!')

    else:
        st.warning("Please upload a CSV file to proceed.")


    
# Check the selected page and call the corresponding function
if page == "Twitter Interaction":
    twitter_interaction_page()
elif page == "Custom Twitter Interaction":
    custom_twitter_interaction_page()

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



