# Adapted form https://github.com/dibollinger/CookieBlock-Consent-Classifier
"""
Using a pretrained model, and given cookie data in JSON format, predict labels for each cookie.
Can choose between the three tree boosters.

Usage:
    predict_class <model_path> <json_data>

Options:
    -h --help              Show this help message.
"""

import json
import logging
import os
import sys
from typing import Dict, Optional, Union

import numpy as np
import xgboost as xgb
from docopt import docopt
from scipy.sparse import csr_matrix

from classifiers.cookieblock_classifier.feature_extraction import CookieFeatureProcessor

logger = logging.getLogger("classifier")


def get_equal_loss_weights():
    """Replicates the argmax probability decision."""
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 0],
        ]
    )


def bayesian_decision(prob_vectors: np.ndarray, loss_weights: np.ndarray):
    """
    Compute class predictions using Bayesian Decision Theory.
    :param prob_vectors: Probability vectors returns by the multiclass classification.
    :param loss_weights: nclass x nclass matrix, loss per classification choice
    :return: Numpy array of discrete label predictions.
    """
    num_instances, num_classes = prob_vectors.shape
    assert loss_weights.shape == (
        num_classes,
        num_classes,
    ), f"Loss weight matrix shape does not match number of actual classes: {loss_weights.shape} vs. {num_classes} classes"
    b = np.repeat(prob_vectors[:, :, np.newaxis], num_classes, axis=2)
    return np.argmin(np.sum(b * loss_weights, axis=1), axis=1)


class ModelWrapper:
    def __init__(self, model_path: str) -> None:
        self.model: Optional[xgb.Booster] = None
        if model_path.endswith(".xgb"):
            self.model = xgb.Booster(model_file=model_path)
        else:
            error_msg: str = f"Unrecognized model type for '{model_path}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def predict_with_bayes_decision(
        self, data: csr_matrix, loss_function: np.ndarray
    ) -> np.ndarray:
        dmat = xgb.DMatrix(data)
        predicted_probabilities = self.model.predict(dmat, training=False)
        return bayesian_decision(predicted_probabilities, loss_function)


def setup_logger() -> None:
    """Log to standard output"""
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


def predict(cookies_data: Union[Dict, str], model_path: str) -> Dict:
    if isinstance(cookies_data, str):
        with open(cookies_data) as f:
            cookies_data = json.load(f)

    # Set up feature processor and model
    cfp = CookieFeatureProcessor(
        "classifiers/cookieblock_classifier/features.json", skip_cmp_cookies=False
    )
    model = ModelWrapper(model_path)

    logger.info("Extracting Features...")
    cfp.extract_features(cookies_data)
    sparse = cfp.retrieve_sparse_matrix()

    logger.info("Predicting Labels...")
    loss_function = np.array([[0, 1.0], [1.0, 0]])

    discrete_predictions = model.predict_with_bayes_decision(sparse, loss_function)

    assert len(discrete_predictions) == len(cookies_data)

    predictions: Dict[str, int] = dict()
    i: int = 0
    for k in cookies_data.keys():
        predictions[k] = int(discrete_predictions[i])
        i += 1

    cfp.reset_processor()
    return predictions


def main() -> int:
    """
    Run the prediction.
    :return: exit code
    """
    argv = None
    cargs = docopt(__doc__, argv=argv)

    setup_logger()

    cookie_data: Dict[str, Dict]
    data_path = cargs["<json_data>"]
    if os.path.exists(data_path):
        try:
            with open(data_path) as fd:
                cookie_data = json.load(fd)
        except json.JSONDecodeError:
            logger.error("File is not a valid JSON object.")
            return 2
    else:
        logger.error(f"File does not exist: '{data_path}'")
        return 1

    model_path = cargs["<model_path>"]
    if not os.path.exists(model_path):
        logger.error(f"Model does not exist: '{model_path}'")
        return 3

    predictions = predict(cookie_data, model_path)

    with open("predictions.json", "w") as fd:
        json.dump(predictions, fd)
    return 0


if __name__ == "__main__":
    exit(main())
