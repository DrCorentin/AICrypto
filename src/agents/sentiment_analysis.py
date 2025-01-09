# src/agents/sentiment_analysis.py

import requests
import re
from textblob import TextBlob

def fetch_social_data(platform="twitter", keyword="crypto", limit=100):
    """
    Fetch social media data related to the keyword.
    Placeholder for API integration (e.g., Twitter API).
    """
    # Simulated social media data as placeholders
    social_data = [
        "Bitcoin to the moon!",
        "Meme coins are skyrocketing!",
        "Market is crashing. Sell everything!",
        "Crypto adoption is growing!"
    ]
    return social_data[:limit]

def clean_text(text):
    """
    Clean the text by removing URLs, special characters, and numbers.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    return text

def get_sentiment_score(data):
    """
    Calculate sentiment score from a list of social media texts.
    """
    sentiment_scores = []
    for text in data:
        cleaned_text = clean_text(text)
        blob = TextBlob(cleaned_text)
        sentiment_scores.append(blob.sentiment.polarity)
    # Average sentiment score
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
