"""This script can be used to:
 1) pre-process the training data for CookieBlock
 2) evaluate the CookieBlock model on the website level
 3) train the CookieBlock model on the full dataset
 Some code is adapted from https://github.com/dibollinger/CookieBlock-Consent-Classifier
"""
import json
import random
import re
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import plotly.graph_objs as go
import xgboost as xgb
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold
from tqdm import tqdm

from feature_extraction import CookieFeatureProcessor
from shared_utils import read_txt_file, write_to_file
from utils import load_data

featuremap_path = "classifiers/cookieblock_classifier/features.json"


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
    new_url = re.sub("/.*", "", new_url)
    return new_url


def restructure_raw_data_per_domain(
    input_filename: str, output_filename: Optional[str] = None
) -> Dict:
    with open(input_filename, "r") as fin:
        data = json.load(fin)
    restructured_data = dict()
    for key, row in data.items():
        domain = url_to_uniform_domain(row["first_party_domain"])
        if domain not in restructured_data:
            restructured_data[domain] = dict()
        restructured_data[domain][key] = row
    if output_filename is not None:
        with open(output_filename, "w") as fout:
            json.dump(restructured_data, fout)
    return restructured_data


def prepare_training_data(
    input_data: Union[str, Dict],
    output_dir: str,
    skip_cmp_cookies: bool,
    cmps_to_ignore: List[int],
):
    if isinstance(input_data, str):
        with open(input_data, "r") as fin:
            input_data = json.load(fin)
    cookiebot_domains = []
    for domain in tqdm(input_data):
        cmp_origin = list(input_data[domain].values())[-1]["cmp_origin"]
        if cmp_origin in cmps_to_ignore:
            continue
        for cookies in input_data[domain].values():
            cookies["cookie_domain"] = cookies.pop("domain")
            cookies["website"] = cookies.pop("first_party_domain")
        feature_processor = CookieFeatureProcessor(
            featuremap_path, skip_cmp_cookies=skip_cmp_cookies
        )
        feature_processor.extract_features_with_labels(
            input_data[domain], disable_tqdm=True
        )
        if len(feature_processor) > 0:
            Path(output_dir).mkdir(exist_ok=True)
            output_filename = str(Path(output_dir) / f"{domain}.sparse")
            # Sparse is the only conversion format supported
            feature_processor.dump_sparse_matrix(output_filename, dump_weights=False)
            if cmp_origin == 0:
                cookiebot_domains.append(domain)
    write_to_file("\n".join(input_data.keys()), Path(output_dir) / "domains.txt")
    write_to_file("\n".join(cookiebot_domains), Path(output_dir) / "cookiebot_domains.txt")


def load_per_domain_data(data_dir: Union[str, Path], mode: str, data: Dict, sample: bool = False) -> Optional[List[str]]:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    domains = read_txt_file(data_dir / "domains.txt")
    for domain in domains:
        filename = data_dir / f"{domain}.sparse"
        if filename.is_file():
            X, y, W = load_data(str(filename))
            if domain not in data:
                data[domain] = dict()
            data[domain][mode] = X, y, W
    if sample:
        cookiebot_domains = read_txt_file(data_dir / "cookiebot_domains.txt")
        return random.sample(cookiebot_domains, 3000)


def compute_base_metrics(labels, predictions, metrics_dict):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    if "accuracy" not in metrics_dict:
        metrics_dict.update(
            dict(
                accuracy=[accuracy],
                precision=[precision],
                recall=[recall],
                f1=[f1],
            )
        )
    else:
        metrics_dict["accuracy"].append(accuracy)
        metrics_dict["precision"].append(precision)
        metrics_dict["recall"].append(recall)
        metrics_dict["f1"].append(f1)


def unpack_data(
    data: Dict[str, Dict[str, Tuple]],
    domains_sublist: List[str],
    mode: str,
) -> Tuple[csr_matrix, List[int], Optional[List[float]]]:
    X = vstack([data[domain][mode][0] for domain in domains_sublist])
    y = [label for domain in domains_sublist for label in data[domain][mode][1]]
    W = None
    if data[domains_sublist[0]][mode][2] is not None:
        W = [weight for domain in domains_sublist for weight in data[domain][mode][2]]
    return X, y, W


def split_train_crossvalidate(
    data: Dict[str, Dict[str, Tuple]],
    test_domains: Optional[List[str]],
    eval_on_reject_all_data: bool = True,
    output_figures: bool = True,
    apply_weights_to_validation: bool = False,
) -> None:
    """
    Apply the split training as if it were cross-validation.
    """

    params = get_params()

    class_names = ["necessary + functional", "analytics + advertising"]
    num_classes = len(class_names)

    domains = list(data.keys())
    accept_domains = [domain for domain in domains if "accept_all" in data[domain]]
    if test_domains is not None:
        accept_domains = [
            domain for domain in accept_domains if domain not in test_domains
        ]

    counts = [len(data[domain]["accept_all"][1]) for domain in accept_domains]

    kf = KFold(n_splits=5)

    per_cookie_metrics = dict()
    per_domain_metrics = dict()

    for train_indices, test_indices in kf.split(accept_domains):
        train_domains = [
            domain for idx, domain in enumerate(accept_domains) if idx in train_indices
        ]
        val_domains = [
            domain for idx, domain in enumerate(accept_domains) if idx in test_indices
        ]

        X_train, y_train, w_train = unpack_data(data, train_domains, "accept_all")
        X_val, y_val, w_val = unpack_data(data, val_domains, "accept_all")
        dtrain: xgb.DMatrix = xgb.DMatrix(data=X_train, label=y_train, weight=w_train)

        dtest: xgb.DMatrix
        if apply_weights_to_validation:
            dtest = xgb.DMatrix(data=X_val, label=y_val, weight=w_val)
        else:
            dtest = xgb.DMatrix(data=X_val, label=y_val)

        bst: xgb.Booster = xgb.train(
            dict(n_jobs=2, random_state=0, **params),
            dtrain,
            100,
        )

        # Evaluate on the "accept all" data
        predicted_probabilities: np.ndarray = bst.predict(
            dtest, ntree_limit=0, training=False
        )
        predictions = np.argmax(predicted_probabilities, axis=1)
        labels: np.ndarray = np.array(y_val).astype(int)

        compute_base_metrics(labels, predictions, per_cookie_metrics)

        # Evaluate on the "reject all" data
        if eval_on_reject_all_data:
            if test_domains is None:
                test_reject_domains = [
                    domain
                    for domain in domains
                    if "reject_all" in data[domain] and domain not in train_domains
                ]
            else:
                test_reject_domains = test_domains
                assert all("reject_all" in data[domain] for domain in test_domains)

            X_test_r, y_test_r, w_test_r = unpack_data(
                data, test_reject_domains, "reject_all"
            )
            dtest_r: xgb.DMatrix
            if apply_weights_to_validation:
                dtest_r = xgb.DMatrix(data=X_test_r, label=y_test_r, weight=w_test_r)
            else:
                dtest_r = xgb.DMatrix(data=X_test_r, label=y_test_r)
            predictions = np.argmax(
                bst.predict(dtest_r, ntree_limit=0, training=False), axis=1
            )
            labels = np.array(y_test_r).astype(int)
            per_domain_predictions = []
            per_domain_labels = []
            per_domain_num_cookies = []
            cur = 0
            for idx, domain in enumerate(test_reject_domains):
                num_cookies = len(data[domain]["reject_all"][1])
                per_domain_predictions.append(
                    predictions[cur : cur + num_cookies].sum()
                )
                per_domain_labels.append(labels[cur : cur + num_cookies].sum())
                cur += num_cookies
                per_domain_num_cookies.append(num_cookies)

            per_domain_labels_aggr = (np.array(per_domain_labels) > 0).astype(int)
            for threshold in range(3):
                per_domain_metrics[threshold] = dict()
                per_domain_predictions_aggr = (
                    np.array(per_domain_predictions) > threshold
                ).astype(int)
                compute_base_metrics(
                    per_domain_labels_aggr,
                    per_domain_predictions_aggr,
                    per_domain_metrics[threshold],
                )

            if "predictions" in per_domain_metrics:
                per_domain_metrics["predictions"].extend(per_domain_predictions)
                per_domain_metrics["labels"].extend(per_domain_labels)
            else:
                per_domain_metrics["predictions"] = per_domain_predictions
                per_domain_metrics["labels"] = per_domain_labels

    for metrics_dict, msg in zip(
        [per_cookie_metrics, *[per_domain_metrics[k] for k in range(3)]],
        [
            "Evaluation on the cookie level (not included in the paper):",
            *[
                f"Evaluation on the website level after clicking on reject (threshold = {k+1})"
                for k in range(3)
            ],
        ],
    ):
        print(
            f"\n{msg}",
        )
        for key, value in metrics_dict.items():
            print(f"{key}: {np.mean(value):.4f}")

    MAX_PRED = 100
    aggr_predictions = [0] * (MAX_PRED + 1)
    aggr_counts = [0] * (MAX_PRED + 1)
    for prediction, label in sorted(
        zip(per_domain_metrics["predictions"], per_domain_metrics["labels"])
    ):
        prediction = min(prediction, MAX_PRED)
        aggr_counts[prediction] += 1
        aggr_predictions[prediction] += prediction > 0 and label > 0

    cumulative_accuracy = [0] * (MAX_PRED + 1)
    cumulative_predictions = [0] * (MAX_PRED + 1)
    cumulative_counts = [0] * (MAX_PRED + 1)
    for idx in range(100, -1, -1):
        cumulative_predictions[idx] = (
            cumulative_predictions[idx + 1] if idx < MAX_PRED else 0
        ) + aggr_predictions[idx]
        cumulative_counts[idx] = (
            cumulative_counts[idx + 1] if idx < MAX_PRED else 0
        ) + aggr_counts[idx]
        cumulative_accuracy[idx] = (
            cumulative_predictions[idx] / cumulative_counts[idx]
            if cumulative_counts[idx] > 0
            else 0.0
        )

    if output_figures:
        figs = [go.Figure()]
        accuracies = [np.mean(per_domain_metrics[k]["accuracy"]) for k in range(10)]
        precisions = [np.mean(per_domain_metrics[k]["precision"]) for k in range(10)]
        recalls = [np.mean(per_domain_metrics[k]["recall"]) for k in range(10)]
        f1_scores = [np.mean(per_domain_metrics[k]["f1"]) for k in range(10)]
        figs[-1].add_trace(
            go.Scatter(
                name="accuracy", x=list(range(1, 11)), y=accuracies, legendrank=1
            )
        )
        figs[-1].add_trace(
            go.Scatter(
                name="precision", x=list(range(1, 11)), y=precisions, legendrank=2
            ),
        )
        figs[-1].add_trace(
            go.Scatter(name="recall", x=list(range(1, 11)), y=recalls, legendrank=3),
        )
        figs[-1].add_trace(
            go.Scatter(name="f1", x=list(range(1, 11)), y=f1_scores, legendrank=4),
        )
        figs[-1].update_layout(
            title="Evaluation metrics of the model predicting whether a website uses AA cookies after clicking on Reject (model trained on OneTrust and Termly)",
            hovermode="x",
        )
        figs[-1].show()


def get_params() -> Dict[str, Union[None, float, int, str]]:
    """
    Best training parameters found so far, for simple CV/split training.
    """
    return {
        "booster": "gbtree",
        "verbosity": 1,
        "nthread": int(cpu_count() - 1),
        # Tree parameters
        "learning_rate": 0.1,
        "gamma": 1,
        "max_depth": 12,
        "min_child_weight": 1,
        "max_delta_step": 0,
        "subsample": 1,
        "sampling_method": "uniform",
        "lambda": 1,
        "alpha": 2,
        "grow_policy": "depthwise",
        "max_leaves": 0,
        "max_bin": 1024,
        "predictor": "auto",
        # Learning Task Parameters
        "objective": "multi:softprob",
        "num_class": 2,
        "base_score": 0.2,
    }


def train_on_full_dataset(data: Dict[str, Dict[str, Tuple]]) -> None:
    domains = [domain for domain in data.keys() if "accept_all" in data[domain]]
    X, y, weights = unpack_data(data, domains, "accept_all")

    dtrain: xgb.DMatrix = xgb.DMatrix(data=X, label=y, weight=weights)
    params = get_params()

    bst: xgb.Booster = xgb.train(
        dict(n_jobs=2, random_state=0, **params),
        dtrain,
        num_boost_round=100,
    )

    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    bst.save_model(f"models/cookieblock_{now_str}.xgb")


@click.command()
@click.option("--crawled_data", type=str, default=None)
@click.option("--output_dir", type=str, default=None)
@click.option("--accept_all_dir", type=str)
@click.option("--reject_all_dir", type=str, default=None)
@click.option("--prepare_data", is_flag=True)
@click.option("--train", is_flag=True)
@click.option("--ignore_cmp", multiple=True, type=int)
def main(
    crawled_data: str,
    output_dir: str,
    accept_all_dir: str,
    reject_all_dir: Optional[str],
    prepare_data: bool,
    train: bool,
    ignore_cmp: Tuple[int],
):
    if prepare_data:
        assert crawled_data is not None and output_dir is not None
        restructured_data = restructure_raw_data_per_domain(crawled_data)
        prepare_training_data(restructured_data, output_dir, skip_cmp_cookies=True, cmps_to_ignore=list(ignore_cmp))
    else:
        data = dict()
        load_per_domain_data(accept_all_dir, "accept_all", data)
        test_domains = None
        if reject_all_dir is not None:
            test_domains = load_per_domain_data(reject_all_dir, "reject_all", data, sample=True)
        if train:
            train_on_full_dataset(data)
        else:
            split_train_crossvalidate(data, test_domains, output_figures=False)


if __name__ == "__main__":
    random.seed(0)
    main()
