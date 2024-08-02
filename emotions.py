import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
model = pickle.load(open(r"C:\Users\HP\AI ELITE 20\ML WEEKEND PROJECTS\nb_emotion.pkl", 'rb'))

# Load the CountVectorizer used for training
with open(r"C:\Users\HP\AI ELITE 20\bow_emotion.pkl", 'rb') as f:
    bow = pickle.load(f)

st.image(r"C:\Users\HP\OneDrive\Pictures\inoopng.png")
st.title("Emotion Analysis System")

# Input email text
text = st.text_input("enter the text prompt")

if text:
    # Transform the input email text to feature array
    data = bow.transform([text]).toarray()

    # Predict if the email is spam or ham
    op = model.predict(data)[0]


    
# Display the prediction when the button is pressed
    if st.button('Submit'):
        
        if op == 0 :
            st.write("SAD")
            st.image(r"C:\Users\HP\Downloads\sad_emoji.png")
        elif op == 1 :
            st.write("JOY")
            st.image(r"C:\Users\HP\Downloads\joy_emoji.png")
        elif op == 2 :
            st.write("LOVE")
            st.image(r"C:\Users\HP\Downloads\love_emoji.png")
        elif op == 3 :
            st.write("ANGER")
            st.image(r"C:\Users\HP\Downloads\Angry_emoji.png")
        elif op == 4 :
            st.write("FEAR")
            st.image(r"C:\Users\HP\Downloads\fear_emoji.png")
        elif op == 5 :
            st.write("SURPRISE")
            st.image(r"C:\Users\HP\Downloads\surprize_emoji.png")
            