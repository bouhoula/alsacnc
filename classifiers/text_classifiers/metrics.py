from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor
from transformers.trainer_utils import EvalPrediction


def get_prediction(
    logits: Union[np.ndarray, Tensor],
    do_multi_label_classification: bool = False,
    task: Optional[str] = None,
) -> Union[np.ndarray, Tensor]:
    if do_multi_label_classification:
        if task in ["ie_text_classification", "ie_text_ps_classification"]:
            logits = np.array(logits)
            predictions = np.zeros_like(logits)
            indices = np.expand_dims(np.argmax(logits, axis=1), axis=1)
            np.put_along_axis(predictions, indices, 1.0, axis=1)
        else:
            threshold = 0.5
            predictions = expit(logits)
            predictions[predictions < threshold] = 0
            predictions[predictions >= threshold] = 1
    else:
        predictions = np.argmax(logits, axis=-1)
    return predictions


def get_metric_fn(
    do_multi_label_classification: bool = False,
    task: Optional[str] = None,
) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics(pred: EvalPrediction) -> Dict:
        labels = pred.label_ids
        predictions = get_prediction(
            pred.predictions, do_multi_label_classification, task
        )
        if do_multi_label_classification:
            cm = multilabel_confusion_matrix(labels, predictions)
        else:
            cm = confusion_matrix(labels, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        acc = accuracy_score(labels, predictions)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
        }

    return compute_metrics
