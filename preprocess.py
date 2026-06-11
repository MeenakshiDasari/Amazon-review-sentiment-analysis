import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # tokenize
    words = word_tokenize(text)

    # remove stopwords and lemmatize
    cleaned_words = []

    for word in words:

        if word not in stop_words:
            cleaned_words.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(cleaned_words)