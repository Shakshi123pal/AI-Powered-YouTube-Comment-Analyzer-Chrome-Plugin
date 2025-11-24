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

# Load parameters from params.yaml
def load_params():
    try:
        params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"
        )
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Loaded params from params.yaml")
        return params
    except Exception as e:
        logger.error(f"Error loading params.yaml: {e}")
        raise

def preprocess_comment(comment, pp):
    """Clean & normalize text using params.yaml controls."""
    try:
        # lowercase
        if pp["lowercase"]:
            comment = comment.lower().strip()

        # remove punctuation
        if pp["remove_punctuation"]:
            comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        # stopwords
        if pp["remove_stopwords"]:
            stop_words = set(stopwords.words("english")) - {
                "not",
                "but",
                "however",
                "no",
                "yet",
            }
            comment = " ".join([w for w in comment.split() if w not in stop_words])

        # lemmatization
        if pp["apply_lemmatization"]:
            lemmatizer = WordNetLemmatizer()
            comment = " ".join([lemmatizer.lemmatize(w) for w in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return comment

def normalize_text(df, pp):
    """Apply preprocessing to dataframe."""
    try:
        # Agar clean_comment column nahi hai, to Comment se banao
        if "clean_comment" not in df.columns:
            if "Comment" in df.columns:
                df["clean_comment"] = df["Comment"].astype(str)
            else:
                raise KeyError("Neither 'clean_comment' nor 'Comment' column found")

        df["clean_comment"] = df["clean_comment"].astype(str)
        df["clean_comment"] = df["clean_comment"].apply(
            lambda c: preprocess_comment(c, pp)
        )

        logger.debug("Applied text normalization")
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

        logger.debug("Saved processed datasets into data/interim")
    except Exception as e:
        logger.error(f"Saving error: {e}")
        raise

def main():
    try:
        logger.debug("=== Starting Preprocessing Stage ===")

        # Load parameters
        params = load_params()
        pp = params["data_preprocessing"]

        # Load input train-test (from ingestion stage)
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
        logger.debug("Loaded train/test datasets")

        # Normalize text
        train_processed = normalize_text(train_df, pp)
        test_processed = normalize_text(test_df, pp)

        # Save output
        save_data(train_processed, test_processed)

        logger.debug("=== Preprocessing Completed Successfully ===")

    except Exception as e:
        logger.error(f"Preprocessing FAILED: {e}")
        print("Error:", e)

if __name__ == "__main__":
    main()
