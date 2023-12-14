import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from PIL import Image
import time

# ... (rest of the code remains unchanged)

# Streamlit UI
st.image(logo, caption=None, width=10, use_column_width=True)
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Share your thoughts:", "", key="user_input")

# Button to trigger analysis
analyze_button = st.button("Analyze")

# View flag for detailed predictions
view_predictions = st.checkbox("View Detailed Predictions", value=False)

# Placeholder for success message
success_placeholder = st.empty()

# Check if the user has entered any text and the button is clicked
if user_input and analyze_button:
    # Make binary prediction and check for offensive words
    binary_result, offensive_words = binary_cyberbullying_detection(user_input)
    st.markdown("<div class='st-bw'>", unsafe_allow_html=True)
    
    if view_predictions:
        st.write(f"Binary Cyberbullying Prediction: {'Cyberbullying' if binary_result == 1 else 'Not Cyberbullying'}")

    # Display offensive words and provide recommendations
    if offensive_words and view_predictions:
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
            st.error(f"Please edit your tweet before resending. Your text contains content that may appear as bullying to other users. {predicted_class.replace('_', ' ').title()}.")
        elif offensive_words and not view_predictions:
            st.warning("While this tweet is not necessarily cyberbullying, it may contain offensive language. Consider editing.")
        else:
            # Display success message before sending
            success_placeholder.success('This tweet is safe to send.')

            # Set a timer for the success message
            for seconds in range(5, 0, -1):
                time.sleep(1)
                success_placeholder.text(f'Tweet Sent! Redirecting in {seconds} seconds...')

            # Reset the success message after the timer
            success_placeholder.empty()

            # Simulate redirecting or any other action after the timer
            st.success('Tweet Sent!')
