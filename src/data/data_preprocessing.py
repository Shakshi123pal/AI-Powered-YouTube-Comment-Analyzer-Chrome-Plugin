import os
import re
import pandas as pd
import nltk
import yaml
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download NLTK resources
nltk.download("wordnet")
nltk.download("stopwords")


def load_params():
    """Load parameters from params.yaml."""
    try:
        params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"
        )
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logger.error(f"Error loading params.yaml: {e}")
        raise


def remove_url(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def create_sentiment(comment):
    """Rule-based sentiment label (0=neutral, 1=positive, 2=negative)."""
    comment = comment.lower()

    positive_words = ["love", "like", "great", "good", "amazing", "nice", "wow"]
    negative_words = ["hate", "bad", "fuck", "shit", "worst", "angry"]

    if any(word in comment for word in positive_words):
        return 1
    if any(word in comment for word in negative_words):
        return 2
    return 0


def preprocess_comment(comment, pp):
    """Apply NLP preprocessing steps."""
    try:
        if pd.isna(comment):
            return ""

        # Remove URLs
        comment = remove_url(comment)

        # Lowercase
        if pp["lowercase"]:
            comment = comment.lower()

        # Remove extra spaces
        comment = comment.strip()

        # Remove punctuation
        if pp["remove_punctuation"]:
            comment = re.sub(r"[^a-zA-Z0-9\s]", "", comment)

        # Remove stopwords
        if pp["remove_stopwords"]:
            stop_words = set(stopwords.words("english")) - {
                "not",
                "but",
                "no",
                "yet",
                "however",
            }
            comment = " ".join(w for w in comment.split() if w not in stop_words)

        # Lemmatization
        if pp["apply_lemmatization"]:
            lemma = WordNetLemmatizer()
            comment = " ".join(lemma.lemmatize(w) for w in comment.split())

        return comment

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return ""


def normalize_text(df, pp):
    """Clean dataframe completely."""
    try:
        # Keep only necessary columns
        needed_cols = ["Comment"]
        df = df[needed_cols]

        # Remove NAN & empty
        df.dropna(subset=["Comment"], inplace=True)

        # Remove duplicates
        df.drop_duplicates(subset=["Comment"], inplace=True)

        # Clean Comment → clean_comment
        df["clean_comment"] = df["Comment"].astype(str)
        df["clean_comment"] = df["clean_comment"].apply(lambda x: preprocess_comment(x, pp))

        # Remove empty cleaned rows
        df = df[df["clean_comment"].str.strip() != ""]

        # Create sentiment label
        df["category"] = df["clean_comment"].apply(create_sentiment)

        logger.debug("Text cleaning + sentiment labeling completed")
        return df

    except Exception as e:
        logger.error(f"Normalization error: {e}")
        raise


def save_data(train_df, test_df):
    try:
        output_dir = os.path.join("data", "interim")
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

        logger.debug("Saved cleaned datasets into data/interim")
    except Exception as e:
        logger.error(f"Saving error: {e}")
        raise


def main():
    try:
        logger.debug("=== PREPROCESSING START ===")
        params = load_params()
        pp = params["data_preprocessing"]

        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        train_clean = normalize_text(train_df, pp)
        test_clean = normalize_text(test_df, pp)

        save_data(train_clean, test_clean)
        logger.debug("=== PREPROCESSING SUCCESS ===")

    except Exception as e:
        logger.error(f"Preprocessing FAILED: {e}")
        print("Error:", e)


if __name__ == "__main__":
    main()
