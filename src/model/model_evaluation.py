import os
import pickle
import logging
import yaml
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Logging
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
file = logging.FileHandler("model_evaluation_errors.log")
file.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)


def get_root():
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current, "../../"))


def load_test_data():
    root = get_root()
    try:
        df = pd.read_csv(os.path.join(root, "data/interim/test_processed.csv"))
        logger.debug("Loaded test dataset")
        return df
    except Exception as e:
        logger.error(f"TEST DATA LOAD ERROR: {e}")
        raise


def load_model_and_vectorizer():
    root = get_root()
    try:
        with open(os.path.join(root, "svm_model.pkl"), "rb") as f:
            model = pickle.load(f)

        with open(os.path.join(root, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        logger.debug("Loaded model and vectorizer")
        return model, vectorizer

    except Exception as e:
        logger.error(f"MODEL LOAD ERROR: {e}")
        raise


def evaluate_model():
    try:
        df = load_test_data()
        model, vectorizer = load_model_and_vectorizer()

        X_test = vectorizer.transform(df["clean_comment"])
        y_test = df["category"].values

        preds = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        logger.debug("Evaluation metrics calculated")

        # Save results directory
        root = get_root()
        reports_dir = os.path.join(root, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(reports_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")

        # Save classification report
        with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(y_test, preds))

        # Save confusion matrix
        cm = confusion_matrix(y_test, preds)
        pd.DataFrame(cm).to_csv(os.path.join(reports_dir, "confusion_matrix.csv"),
                                index=False)

        logger.debug("Evaluation reports saved")

    except Exception as e:
        logger.error(f"MODEL EVALUATION FAILED: {e}")
        print("Error:", e)


def main():
    logger.debug("=== MODEL EVALUATION START ===")
    evaluate_model()
    logger.debug("=== MODEL EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
