import logging
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    scaler = StandardScaler()

    clf_params = {"C": float("inf"), "max_iter": 500}

    clf = LogisticRegression(**clf_params, random_state=SEED)

    clf_key = "clf"
    pipeline = Pipeline([("scaler", scaler), (clf_key, clf)])

    data_params: dict[str, Any] = {
        "n_samples": 1000,
        "n_features": 10,
        "n_informative": 5,
    }

    logger.info("Pipeline and data parameters defined")

    X, y = make_classification(**data_params, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    logger.info("Data split")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    experiment_name = "sklearn-make-classification"
    mlflow.set_experiment(experiment_name)

    logger.info("Connection and experiment with name %s initialized", experiment_name)

    with mlflow.start_run():
        mlflow.log_params(data_params)
        mlflow.log_params({f"{clf_key}__{key}": val for key, val in clf_params.items()})
        mlflow.log_param("random_seed", SEED)
        logger.info("Parameters logged")

        pipeline.fit(X_train, y_train)
        logger.info("Pipeline fitted on %d samples", len(X_train))

        y_pred = pipeline.predict(X_test)
        test_precision = precision_score(y_test, y_pred)

        mlflow.log_metric("precision", float(test_precision))
        logger.info("Test Precision: %.4f logged", test_precision)

        with TemporaryDirectory() as tmp_dir:
            model_path = f"{tmp_dir}/sklearn-make-classification.joblib"
            dump(pipeline, model_path)
            mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
