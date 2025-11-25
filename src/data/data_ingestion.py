import os
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load YAML parameters."""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug(f"Loaded params from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        raise


def load_data(csv_path: str) -> pd.DataFrame:
    """Load pre-downloaded YouTube comments CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.debug(f"Loaded data from: {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for YouTube comments."""
    try:
        # Drop missing rows
        df.dropna(inplace=True)

        # Ensure Comment is string
        df["Comment"] = df["Comment"].astype(str)

        # Remove empty comments
        df = df[df["Comment"].str.strip() != ""]

        # Remove duplicates
        df.drop_duplicates(subset=["Comment"], inplace=True)

        logger.debug("Preprocessing complete.")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise


def save_data(train_df, test_df, base_dir: str):
    """Save final train/test CSVs inside data/processed."""
    try:
        processed_path = os.path.join(base_dir, "processed")
        os.makedirs(processed_path, exist_ok=True)

        train_df.to_csv(os.path.join(processed_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(processed_path, "test.csv"), index=False)

        logger.debug(f"Saved processed data to {processed_path}")

    except Exception as e:
        logger.error(f"Failed saving processed data: {e}")
        raise


def main():
    try:
        # Load params.yaml
        params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../params.yaml"
        )
        params = load_params(params_path)
        test_size = params["data_ingestion"]["test_size"]

        # Ensure raw folder exists
        raw_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/raw"
        )
        os.makedirs(raw_dir, exist_ok=True)

        # Load CSV from the raw folder
        raw_data_path = os.path.join(raw_dir, "all_comments.csv")
        df = load_data(raw_data_path)

        # Clean the data
        df = preprocess_data(df)

        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        # Save processed data
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
        save_data(train_df, test_df, base_dir)

        logger.debug("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        print(" Error:", e)



if __name__ == "__main__":
    main()
