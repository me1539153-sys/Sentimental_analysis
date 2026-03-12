import streamlit as st
from transformers import pipeline
import os

st.title("Human Emotion Detector")

@st.cache_resource
def load_local_model():
    # This path points to the folder you just downloaded from Colab
    model_path = r"C:\Users\hp\Desktop\ml_mini_project\sentiment.py\my_emotion_model"
    
    # We load the local model and local feature extractor together
    return pipeline("audio-classification", model=model_path, feature_extractor=model_path)
   
with st.spinner("Initializing local AI..."):
    classifier = load_local_model()

if classifier:
    st.success("AI is active (Offline Mode)")
    audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
    
    if audio_file:
        st.audio(audio_file)
        
        # This button MUST be indented so it only shows AFTER a file is uploaded
        if st.button("Detect Emotion"):
            with st.spinner("AI is thinking..."):
                # Use getvalue() to read the file bytes correctly
                audio_bytes = audio_file.getvalue()
                results = classifier(audio_bytes)
                
                st.subheader("Detected Emotions:")
                for res in results:
                    # Display label and percentage
                    st.write(f"**{res['label']}**: {round(res['score']*100, 2)}%")


