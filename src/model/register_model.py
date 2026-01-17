import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

# MLflow Tracking URI (DAGSHUB)

DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
if not DAGSHUB_REPO:
    raise EnvironmentError("DAGSHUB_REPO environment variable not set")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_REPO}.mlflow"
)


# logging configuration

logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load experiment info
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        logger.error('Error loading model info: %s', e)
        raise


# register model
def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}"   # 👈 ONLY CHANGE

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(
            "Model %s v%s registered and promoted to Staging",
            model_name,
            model_version.version
        )

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise



# main
def main():
    try:
        model_info = load_model_info("experiment_info.json")
        register_model("yt_chrome_plugin_model", model_info)
    except Exception as e:
        logger.error("Model registration failed: %s", e)
        print(f"ERROR: {e}")

if __name__ == '__main__':
    main()
