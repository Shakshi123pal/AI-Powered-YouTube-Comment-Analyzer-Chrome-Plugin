import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# logging
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
file = logging.FileHandler("model_building_errors.log")
file.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)


def get_root():
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current, "../../"))


def load_params():
    try:
        root = get_root()
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logger.error("PARAM LOAD ERROR: %s", e)
        raise


def load_data():
    """Load preprocessed train data"""
    root = get_root()
    try:
        df = pd.read_csv(os.path.join(root, "data/interim/train_processed.csv"))
        logger.debug("Loaded processed training data")

        # Drop unused columns if present
        drop_cols = ["Comment"]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # clean_comment must be string
        df["clean_comment"] = df["clean_comment"].fillna("").astype(str)

        return df

    except Exception as e:
        logger.error(f"Data load error: {e}")
        raise


def apply_tfidf(df, max_features, ngram_range):
    """Vectorize cleaned text safely"""
    try:
        df["clean_comment"] = df["clean_comment"].fillna("").astype(str)

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        X = vectorizer.fit_transform(df["clean_comment"])
        y = df["category"].values

        root = get_root()
        with open(os.path.join(root, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        return X, y

    except Exception as e:
        logger.error(f"TF-IDF ERROR: {e}")
        raise


def train_svm(X, y, params):
    """Train SVM with automatic class balancing"""
    try:
        model = SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            probability=True,
            class_weight="balanced"  # VERY IMPORTANT FOR YOUR DATA
        )
        model.fit(X, y)
        logger.debug("SVM training completed")
        return model

    except Exception as e:
        logger.error(f"SVM TRAIN ERROR: {e}")
        raise


def save_model(model):
    root = get_root()
    try:
        with open(os.path.join(root, "svm_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved successfully")

    except Exception as e:
        logger.error(f"Model save error: {e}")
        raise


def main():
    try:
        logger.debug("=== MODEL BUILDING START ===")

        params = load_params()
        svm_params = params["model_svm"]

        df = load_data()

        X, y = apply_tfidf(
            df,
            max_features=params["model_building"]["max_features"],
            ngram_range=tuple(params["model_building"]["ngram_range"])
        )

        model = train_svm(X, y, svm_params)

        save_model(model)

        logger.debug("=== MODEL BUILDING SUCCESS ===")

    except Exception as e:
        logger.error(f"MODEL BUILD FAILED: {e}")
        print("Error:", e)


if __name__ == "__main__":
    main()
