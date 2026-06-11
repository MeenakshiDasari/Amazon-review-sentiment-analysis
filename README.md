# Amazon Review Sentiment Analysis

A Machine Learning project that predicts whether an Amazon product review is Positive or Negative.

## Features
- Text preprocessing using NLTK
- TF-IDF Vectorization
- Logistic Regression Classifier
- Streamlit Web Application
- Real-time Sentiment Prediction

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit

## Project Structure
Amazon-review-sentiment-analysis/
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf.pkl
│
├── app.py
├── train.py
├── preprocess.py
├── requirements.txt
└── .gitignore

## Run Locally

pip install -r requirements.txt

streamlit run app.py
