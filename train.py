import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
# Load dataset
df = pd.read_csv("dataset/Reviews.csv")

# Keep required columns
df = df[["Score", "Text"]]

# Create sentiment column
def get_sentiment(score):

    if score >= 4:
        return 1

    elif score <= 2:
        return 0

    else:
        return None

df["sentiment"] = df["Score"].apply(get_sentiment)

# Remove neutral reviews
df = df.dropna()

# Remove empty reviews
df = df.dropna(subset=["Text"])

# Rename column
df = df.rename(columns={"Text": "review"})

# Keep only required columns
df = df[["review", "sentiment"]]
# Split Dataset


# Take sample
df = df.sample(20000, random_state=42)

print("Before Cleaning:")
print(df["review"].iloc[0])

# Apply NLP preprocessing
df["clean_review"] = df["review"].apply(clean_text)
# TF-IDF

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["clean_review"])

y = df["sentiment"]
# Apply NLP preprocessing
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
# Train Model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully")
# Accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nTF-IDF Shape:")
print(X.shape)

print("\nAfter Cleaning:")
print(df["clean_review"].iloc[0])

print("\nShape:")
print(df.shape)


# Save Model
pickle.dump(model, open("model/sentiment_model.pkl", "wb"))

# Save TF-IDF
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))

print("Model Saved Successfully")