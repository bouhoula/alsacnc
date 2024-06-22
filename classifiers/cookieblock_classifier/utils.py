# Adapted form https://github.com/dibollinger/CookieBlock-Consent-Classifier

""" Contains static utility functions for the feature transformation process"""

import base64
import csv
import json
import logging
import os
import pickle
import re
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import js2py
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("classifier")

MIN_B64_LEN: int = 8

# Types accepted by the function below
VariableCookieData = List[Dict[str, Union[int, str]]]


def save_validation(
    dtest: csr_matrix, y_test: List[int], eval_path, timestamp_str
) -> None:
    """
    Save validation data.
    :param dtest: Validation split.
    :param y_test: Corresponding labels.
    :param eval_path: Path to store the files in
    :param timestamp_str: Timestamp for the filename
    """
    dtest_name = os.path.join(eval_path, f"validation_matrix_{timestamp_str}.sparse")
    with open(dtest_name, "wb") as fd:
        pickle.dump(dtest, fd, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Dumped Validation DMatrix to: {dtest_name}")

    labels_fn = dtest_name + ".labels"
    with open(labels_fn, "wb") as fd:
        pickle.dump(y_test, fd, pickle.HIGHEST_PROTOCOL)


def load_data(dpath: str):
    """Multi purpose data loading function.
    Supports loading from xgb binary, libsvm and pickled sparse matrix.
    The filetype is determined from the extension in the given path string.
    :param dpath: Path for the data to be loaded.
    :return a tuple of (features, labels, weights), where 'features' is a sparse matrix and the others are lists.
    """
    features: Union[xgb.DMatrix, csr_matrix]
    labels: Optional[List[float]] = None
    weights: Optional[List[float]] = None

    if dpath.endswith(".buffer"):
        logger.info("Loading DMatrix data...")
        features = xgb.DMatrix(dpath)

    elif dpath.endswith(".libsvm"):
        logger.info("Loading LibSVM data...")
        features, labels = load_svmlight_file(dpath)
        weights_fn = dpath + ".weights"
        if os.path.exists(weights_fn):
            with open(weights_fn, "rb") as fd:
                weights = pickle.load(fd)

    elif dpath.endswith(".sparse"):
        logger.info("Loading sparse data...")
        with open(dpath, "rb") as fd:
            features = pickle.load(fd)

        labels_fn = dpath + ".labels"
        if os.path.exists(labels_fn):
            with open(labels_fn, "rb") as fd:
                labels = pickle.load(fd)

        weights_fn = dpath + ".weights"
        if os.path.exists(weights_fn):
            with open(weights_fn, "rb") as fd:
                weights = pickle.load(fd)
    else:
        logger.error("Unknown data input format.")
        return None

    logger.info("Loading complete.")

    return features, labels, weights


def url_to_uniform_domain(url: str) -> str:
    """
    Takes a URL or a domain string and transforms it into a uniform format.
    Examples: {"www.example.com", "https://example.com/", ".example.com"} --> "example.com"
    :param url: URL to clean and bring into uniform format
    """
    new_url = url.strip()
    new_url = re.sub("^http(s)?://", "", new_url)
    new_url = re.sub("^www([0-9])?", "", new_url)
    new_url = re.sub("^\\.", "", new_url)
    new_url = re.sub("/$", "", new_url)
    return new_url


def load_lookup_from_csv(csv_source: str, count: int) -> Dict[str, int]:
    """
    Load a lookup dictionary, mapping string to rank, from a CSV file.
    Assumes the CSV to be pre-sorted.
    :param csv_source: Source data filepath
    :param count: number of strings to load
    :return: dictionary mapping strings to rank
    """
    lookup_dict: Dict[str, int] = dict()
    rank: int = 0
    with open(csv_source, "r") as fd:
        line = next(fd)
        try:
            while rank < count:
                if line.startswith("#"):
                    line = next(fd)
                    continue
                lookup_dict[line.strip().split(",")[-1]] = rank
                rank += 1
                line = next(fd)
        except StopIteration:
            raise RuntimeError(
                f"Not enough entries in file. Expected at least {count}, max is {rank}."
            )

    return lookup_dict


def check_flag_changed(cookie_updates: VariableCookieData, flag: str) -> bool:
    """
    Checks if given flag (by key) differs between cookie updates.
    If so, return True, else False. May iterate over all updates.
    :param cookie_updates: Cookie update dictionary
    :param flag: Key for the flag to check between updates
    :return: True if differs between at least 1 update, False otherwise.
    """
    v_iter = iter(cookie_updates)
    previous_entry = next(v_iter)  # first entry always non-null
    try:
        while previous_entry is not None:
            next_entry = next(v_iter)
            if next_entry is not None:
                if previous_entry[flag] != next_entry[flag]:
                    return True
            previous_entry = next_entry
    except StopIteration:
        pass
    return False


def try_decode_base64(possible_encoding: str) -> Optional[str]:
    """
    Try to decode the given input object as base64.
    :param possible_encoding: string that is potentially an encoded value
    :return: If result is an UTF-8 string, return it. Else, return None
    """
    if type(possible_encoding) is not str or len(possible_encoding) < MIN_B64_LEN:
        return None
    else:
        try:
            b64decoded = base64.b64decode(possible_encoding)
            return b64decoded.decode("utf-8")
        except (base64.binascii.Error, UnicodeDecodeError, ValueError):
            return None


def try_split_json(possible_json: str) -> Optional[Dict[str, Any]]:
    """
    Try to split the javascript or json string, if possible.
    :param possible_json: string to split into a dictionary
    :return: dictionary containing json keys and attributes
    """
    try:
        return json.loads(possible_json)
    except json.JSONDecodeError:
        try:
            js_func = js2py.eval_js("function a() { return " + possible_json + " }")
            return js_func().to_dict()
        except (
            js2py.internals.simplex.JsException,
            NotImplementedError,
            AttributeError,
        ):
            pass
    return None


def split_delimiter_separated(
    possible_csv: str, csv_sniffer: csv.Sniffer, delimiters: str, min_seps: int = 2
):
    """
    If the given string is delimiter separated, split it and return the list of content strings.
    :param possible_csv: String to split.
    :param csv_sniffer: Sniffer instance to use.
    :param delimiters: String of valid delimiters
    :param min_seps: number of instances required for CSV to be recognized
    :return: None if cannot be separated. Else, a list of strings.
    """
    try:
        dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
        num_separators = possible_csv.count(dialect.delimiter)
        if num_separators > min_seps:
            return list(csv.reader((possible_csv,), dialect))[0], dialect.delimiter
    except csv.Error:
        # not csv formatted -- check if it's base64
        maybe_decoded = try_decode_base64(possible_csv)
        if maybe_decoded is not None:
            try:
                # debug
                # print("Successfully decoded b64 with csv")
                dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
                num_separators = possible_csv.count(dialect.delimiter)
                if num_separators > min_seps:
                    return (
                        list(csv.reader((possible_csv,), dialect))[0],
                        dialect.delimiter,
                    )
            except csv.Error:
                pass

    return None, None


def contains_delimiter_separated(
    possible_csv: str, csv_sniffer: csv.Sniffer, delimiters: str, min_seps: int = 2
) -> bool:
    """
    Verify whether the given string is delimiter separated.
    :param possible_csv: String to verify.
    :param csv_sniffer: Sniffer instance to use.
    :param delimiters: String of valid delimiters
    :param min_seps: number of instances required for CSV to be recognized
    :return: True if delimiter separated, False if not
    """
    try:
        dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
        num_separators = possible_csv.count(dialect.delimiter)
        if num_separators > min_seps:
            return True
    except csv.Error:
        # not csv formatted -- check if it's base64
        maybe_decoded = try_decode_base64(possible_csv)
        if maybe_decoded is not None:
            try:
                # debug
                # print("Successfully decoded b64 with csv")
                dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
                num_separators = possible_csv.count(dialect.delimiter)
                if num_separators > min_seps:
                    return True
            except csv.Error:
                pass

    return False


def delim_sep_check(
    to_check: str, delims: str, min_seps: int
) -> Tuple[Optional[str], int]:
    """
    Determine the best separator in a string, where a separator must appear at least min_seps times.
    Heuristic. Tries to determine if we have CSV separated data.
    :param to_check: String to check
    :param delims: List of delimiters as a string
    :param min_seps: minimum number of occurrences
    :return: Best separator and number of occurrences of this separator.
    """
    maxoccs = min_seps
    chosen_delimiter = None

    for d in delims:
        numoccs = to_check.count(d)
        if numoccs > maxoccs:
            chosen_delimiter = d
            maxoccs = numoccs

    return chosen_delimiter, maxoccs


def log_confidence_per_label(
    probs_with_label: pd.DataFrame,
    class_names: List[str],
    use_true_label: bool,
    comp: Callable,
    logstring: str,
) -> None:
    """
    Prints the confidence for each label to the log.
    This is based on the predicted class probabilities by the classifier.
    Predicted label is computed via a simple argmax probability.
    :param probs_with_label: Pandas dataframe, first column is true labels, rest is probabilities per class.
    :param class_names: Names for each of the classes (and number of classes implicitly)
    :param use_true_label: If true, output confidence for the true label. False, output confidence for the predicted label.
    :param comp: Callable function to decide when to count the confidence (e.g. if true label matches predicted label)
    :param logstring: String prefix to use in the log message.
    """
    num_classes: int = len(class_names)

    # Output mean confidence in proper label to log
    confidence_per_label = [list() for i in range(num_classes)]
    for index, row in probs_with_label.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        if comp(true_label, maxprob_label):
            cur_label = int(row[0]) if use_true_label else maxprob_label
            confidence_per_label[cur_label].append(row[cur_label + 1])

    for i in range(num_classes):
        if len(confidence_per_label[i]) > 1:
            logger.info(
                logstring
                + f"'{class_names[i]}': {mean(confidence_per_label[i]) * 100.0:.3f}%"
                f"+{stdev(confidence_per_label[i]) * 100.0:.3f}%"
            )


def log_accuracy_and_confusion_matrix(
    disc_predictions: np.ndarray, tl_in: np.ndarray, class_names: List[str]
) -> None:
    """
    Log the confusion matrix, and overall accuracy rates for each class inside that confusion matrix.
    Also log the total accuracy over all classes.
    :param disc_predictions: Numpy vector of discrete predictions (i.e. labels)
    :param tl_in: The true label vector, to compute the accuracy.
    :param class_names: Names for the classes.
    """
    num_classes: int = len(class_names)
    num_instances: int = len(disc_predictions)
    assert max(disc_predictions) <= (
        num_classes - 1
    ), "Number of classes in predictions exceeds expected maximum."

    true_labels = np.array([int(t) for t in tl_in])
    pl_list = disc_predictions

    # Compute confusion matrix and output as CSV via pandas
    confusion_matrix: np.ndarray = np.zeros((num_classes, num_classes), dtype=int)

    # Note: this expects labels to be contiguous starting from 0, ending at num_classes, with no gaps in numbering
    for i in range(num_instances):
        confusion_matrix[true_labels[i], pl_list[i]] += 1

    # Output the confusion matrix to the log
    logger.info(f"Confusion Matrix:\n{confusion_matrix}")

    acc_count = accuracy_score(true_labels, pl_list, normalize=False)
    acc_ratio = accuracy_score(true_labels, pl_list, normalize=True)

    logger.info(f"Total Accuracy Count: {acc_count}")
    logger.info(f"Total Accuracy Ratio: {acc_ratio}")

    micro_precision = precision_score(true_labels, pl_list, average="micro")
    micro_recall = recall_score(true_labels, pl_list, average="micro")
    micro_f1score = f1_score(true_labels, pl_list, average="micro")

    logger.info(f"Micro Precision: {micro_precision}")
    logger.info(f"Micro Recall: {micro_recall}")
    logger.info(f"Micro F1Score: {micro_f1score}")

    macro_precision = precision_score(true_labels, pl_list, average="macro")
    macro_recall = recall_score(true_labels, pl_list, average="macro")
    macro_f1score = f1_score(true_labels, pl_list, average="macro")

    logger.info(f"Macro Precision: {macro_precision}")
    logger.info(f"Macro Recall: {macro_recall}")
    logger.info(f"Macro F1Score: {macro_f1score}")

    weighted_precision = precision_score(true_labels, pl_list, average="weighted")
    weighted_recall = recall_score(true_labels, pl_list, average="weighted")
    weighted_f1score = f1_score(true_labels, pl_list, average="weighted")

    logger.info(f"Weighted Precision: {weighted_precision}")
    logger.info(f"Weighted Recall: {weighted_recall}")
    logger.info(f"Weighted F1Score: {weighted_f1score}")

    class_precision = precision_score(true_labels, pl_list, average=None)
    class_recall = recall_score(true_labels, pl_list, average=None)
    class_f1score = f1_score(true_labels, pl_list, average=None)

    logger.info(f"Precision for each class: {class_precision}")
    logger.info(f"Recall for each class: {class_recall}")
    logger.info(f"F1Score for each class: {class_f1score}")

    logger.info("-------------------------------")

    # DISABLED: Output the individual error rates per class
    # for i in range(num_classes):
    #    logger.info(f"Precision errors by true class '{class_names[i]}': {confusion_matrix[:, i] / np.sum(confusion_matrix[:, i])}")
    #    logger.info(f"Recall errors by class '{class_names[i]}': {confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])}")

    # OLD Output Precision + Recall
    precision_vector = np.zeros(num_classes)
    recall_vector = np.zeros(num_classes)
    f1_score_vector = np.zeros(num_classes)
    for i in range(num_classes):
        precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        precision_vector[i] = precision
        recall_vector[i] = recall
        f1_score_vector[i] = 2 * ((precision * recall) / (precision + recall))

    logger.info(
        f"(Old Method) Total Accuracy: {np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) * 100.0:.3f}%"
    )
    logger.info(f"(Old Method) Precision: {precision_vector}")
    logger.info(f"(Old Method) Recall: {recall_vector}")
    logger.info(f"(Old Method) F1 Scores: {f1_score_vector}")


def log_validation_statistics(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    eval_path: str,
    timestamp: str,
) -> None:
    """
    Write the probability predictions to disk and log confidence and confusion matrix.
    :param predicted_probs: N x M matrix, N instances, M classes with probabilities
    :param true_labels: True labels to base accuracy metric off of.
    :param class_names: Names for the classes
    :param eval_path: Path to output files to.
    :param timestamp: timestamp for the filenames.
    """

    # Output the softprob predictions with true labels to a csv first.
    preds_df = pd.DataFrame(predicted_probs, columns=class_names)
    preds_df.insert(0, "labels", true_labels)

    softprob_path = os.path.join(eval_path, f"softprob_predictions_{timestamp}.csv")
    preds_df.to_csv(softprob_path, index=False)
    logger.info(f"Dumped softprob prediction matrix csv to: {softprob_path}")

    # Compute the confidence for 3 different cases:

    # Output confidence for each true label
    log_confidence_per_label(
        preds_df,
        class_names,
        True,
        (lambda t, m: True),
        "Mean/Stddev confidence in true label overall: ",
    )

    # Output confidence where the prediction was correct
    log_confidence_per_label(
        preds_df,
        class_names,
        False,
        (lambda t, m: t == m),
        "Mean/Stddev confidence in predicted label, where prediction is correct: ",
    )

    # Ouput confidence where the prediction was wrong
    log_confidence_per_label(
        preds_df,
        class_names,
        False,
        (lambda t, m: t != m),
        "Mean/Stddev confidence in predicted label, where prediction is incorrect: ",
    )

    logger.info("....................................................................")
    logger.info("Predicted labels & accuracy when using ARGMAX as a prediction rule")

    disc_preds_argmax = np.argmax(predicted_probs, axis=1)
    log_accuracy_and_confusion_matrix(disc_preds_argmax, true_labels, class_names)

    logger.info("....................................................................")


def setupLogger(filename: Optional[str]) -> None:
    """
    Set up the logger instance, which will write its output to stderr.
    :param loglevel: Log level at which to record.
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d-%H:%M:%S",
    )
    ch = logging.StreamHandler()

    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Separator to tell if new log started
    logger.info("=========================================")
