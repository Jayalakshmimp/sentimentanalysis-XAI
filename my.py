import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import nltk
import shap
import torch
import numpy as np

# Download NLTK resources (needed for VADER)
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Load RoBERTa Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function for RoBERTa Sentiment Analysis
def roberta_sentiment(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = "Positive 😊" if scores[2] > scores[0] else "Negative 😢"
    return sentiment, {"Positive": scores[2], "Neutral": scores[1], "Negative": scores[0]}

# Function for VADER Sentiment Analysis
def vader_sentiment(text):
    scores = sia.polarity_scores(text)
    sentiment = "Positive 😊" if scores["compound"] > 0 else "Negative 😢"
    return sentiment, scores

# Function for SHAP Visualization with RoBERTa
def explain_with_shap(text):
    # Define a tokenizer function compatible with SHAP
    def f(x):
        tokens = tokenizer(list(x), return_tensors="pt", padding=True, truncation=True)
        outputs = model(**tokens)
        scores = outputs[0].detach().numpy()
        scores = softmax(scores, axis=1)
        return scores

    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer([text])
    
    return shap_values

# Function for SHAP Visualization for VADER
def explain_vader_with_shap(text):
    def f(x):
        # Return the compound sentiment scores for SHAP input
        return np.array([sia.polarity_scores(t)["compound"] for t in x])
    
    # Create the SHAP explainer with Text masker
    explainer = shap.Explainer(f, masker=shap.maskers.Text())
    shap_values = explainer([text])
    
    return shap_values

# Custom function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

# Streamlit App
st.title("Sentiment Analysis with XAI")
st.write("Analyze the sentiment of text using *RoBERTa* or *VADER* models and explain predictions with SHAP.")

# Text input field
user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type something...")

# Select model
model_choice = st.selectbox("Choose a model:", ("RoBERTa", "VADER"))

# Button to analyze sentiment
if st.button("Analyze"):
    if user_input.strip():
        if model_choice == "RoBERTa":
            # RoBERTa Sentiment Analysis
            sentiment, scores = roberta_sentiment(user_input)
            st.write(f"**RoBERTa Sentiment:** {sentiment}")
            st.write(f"**Scores:** {scores}")
            
            # Generate SHAP explanations for RoBERTa
            st.write("### SHAP Explanation for RoBERTa")
            shap_values = explain_with_shap(user_input)
            st_shap(shap.plots.text(shap_values[0], display=False), height=300)
        
        elif model_choice == "VADER":
            # VADER Sentiment Analysis
            sentiment, scores = vader_sentiment(user_input)
            st.write(f"**VADER Sentiment:** {sentiment}")
            st.write(f"**Scores:** {scores}")
            
            # Generate SHAP explanations for VADER
            st.write("### SHAP Explanation for VADER")
            shap_values = explain_vader_with_shap(user_input)
            st_shap(shap.plots.text(shap_values[0], display=False), height=300)
    else:
        st.warning("Please enter some text to analyze.")
