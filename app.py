import streamlit as st
import pickle

from preprocess import clean_text

# Load Model
model = pickle.load(open("model/sentiment_model.pkl", "rb"))

# Load TF-IDF
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

st.title("Amazon Review Sentiment Analysis")

review = st.text_area("Enter Review")

if st.button("Predict"):

    cleaned = clean_text(review)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Positive 😊")

    else:
        st.error("Negative 😔")