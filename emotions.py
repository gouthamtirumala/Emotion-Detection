import os
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
model = pickle.load(open(r"nb_emotion.pkl", 'rb'))

# Load the CountVectorizer used for training
with open(r"bow_emotion.pkl", 'rb') as f:
    bow = pickle.load(f)

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load images using absolute paths
st.image(os.path.join(current_dir, "inoopng.png"))
st.title("Emotion Analysis System")

# Input email text
text = st.text_input("Enter the text prompt")

if text:
    # Transform the input email text to feature array
    data = bow.transform([text]).toarray()

    # Predict the emotion
    op = model.predict(data)[0]

    # Display the prediction when the button is pressed
    if st.button('Submit'):
        if op == 0:
            st.write("SAD")
            st.image(os.path.join(current_dir, "sad_emoji.png"))
        elif op == 1:
            st.write("JOY")
            st.image(os.path.join(current_dir, "joy_emoji.png"))
        elif op == 2:
            st.write("LOVE")
            st.image(os.path.join(current_dir, "love_emoji.png"))
        elif op == 3:
            st.write("ANGER")
            st.image(os.path.join(current_dir, "Angry_emoji.png"))
        elif op == 4:
            st.write("FEAR")
            st.image(os.path.join(current_dir, "fear_emoji.png"))
        elif op == 5:
            st.write("SURPRISE")
            st.image(os.path.join(current_dir, "surprize_emoji.png"))
