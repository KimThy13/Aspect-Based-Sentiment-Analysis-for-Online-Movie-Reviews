import os
import re
import html
import emoji
import argparse
import warnings
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory, LangDetectException

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Ensure consistent language detection
DetectorFactory.seed = 0

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

warnings.filterwarnings("ignore")

# Regex emoji pattern
EMOJI_PATTERN = re.compile(
    "[" +
    "\U0001F600-\U0001F64F" +
    "\U0001F300-\U0001F5FF" +
    "\U0001F680-\U0001F6FF" +
    "\U0001F700-\U0001F77F" +
    "\U0001F780-\U0001F7FF" +
    "\U0001F800-\U0001F8FF" +
    "\U0001F900-\U0001F9FF" +
    "\U0001FA00-\U0001FA6F" +
    "\U0001FA70-\U0001FAFF" +
    "\U00002702-\U000027B0" +
    "\U000024C2-\U0001F251" +
    "]+", flags=re.UNICODE
)

# Expand common English contractions and informal abbreviations into their full forms
def expand_abbreviations(text):
    abbreviations = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is",
        "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "can't": "cannot", "won't": "will not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "shouldn't": "should not",
        "wouldn't": "would not", "couldn't": "could not", "mustn't": "must not", "shan't": "shall not",
        "let's": "let us", "that's": "that is", "who's": "who is", "what's": "what is", "where's": "where is",
        "there's": "there is", "how's": "how is", "y'all": "you all", "gonna": "going to",
        "wanna": "want to", "gotta": "got to"
    }
    for abb, full in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abb) + r'\b', full, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    """
    Perform text normalization and cleaning for raw review text.
    
    Steps:
        - Remove HTML artifacts
        - Expand abbreviations (contractions and informal forms)
        - Normalize spacing and punctuation
        - Separate digits from letters (e.g., "4k" -> "4 k")
        - Capitalize first letter of each sentence
        - Preserve emojis, remove unwanted symbols

    Returns:
        Cleaned string ready for downstream NLP tasks.
    """

    # Ensure input is a string; otherwise return empty string
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g., &amp; -> &), and trim whitespace
    text = html.unescape(text.strip())

    # Attempt to fix encoding issues (optional fallback)
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Remove any remaining HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Expand common abbreviations (e.g., can't -> cannot)
    text = expand_abbreviations(text)

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove space before punctuation (e.g., "hello !" -> "hello!")
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Add space between numbers and letters (e.g., "4k" -> "4 k")
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

    # Ensure there is a space between end of sentence and next capital word (e.g., "great!Next" -> "great! Next")
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Capitalize the first letter after each sentence-ending punctuation
    def capitalize_sentences(match):
        return match.group(1) + ' ' + match.group(2).upper()

    text = re.sub(r'([.!?])\s+([a-z])', capitalize_sentences, text)

    # Remove unwanted characters, keep only words, numbers, punctuation, and emoji
    text = re.sub(
        r'[^\w\s.,;:!?/"-]+',
        lambda x: x.group() if EMOJI_PATTERN.match(x.group()) else '',
        text
    )

    return text.strip()

def detect_language(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"

def clean_reviews_df(df, text_column="review_text"):
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column] != "No content"]
    df["language"] = df[text_column].apply(detect_language)
    df = df[df["language"] == "en"]
    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column].fillna("").str.strip() != ""]
    df = df.drop_duplicates(subset=[text_column])
    return df.drop(columns=["language"])

def main():
    parser = argparse.ArgumentParser(description="Clean raw movie reviews.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw_reviews.csv")
    parser.add_argument("--output", type=str, default="cleaned_reviews.csv", help="Path to save cleaned output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading raw data from: {args.input}")
    df = pd.read_csv(args.input, index_col=0)
    if "review_text" not in df.columns:
        raise ValueError("Input CSV must contain 'review_text' column")

    df = df[["review_text"]]
    df_cleaned = clean_reviews_df(df)
    df_cleaned.to_csv(args.output, index=False)
    print(f"Saved cleaned data to: {args.output}")

if __name__ == "__main__":
    main()
