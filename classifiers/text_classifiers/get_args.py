from argparse import Namespace
from datetime import datetime
from typing import Any, Callable

import click

from shared_utils import load_yaml


def general_options(f: Callable) -> Callable:
    f = click.option(
        "--config",
        "config_file",
        default="config/purpose_classification_2_labels_config.yaml",
    )(f)
    f = click.option("--tokenizer", default="bert-base-uncased")(f)
    f = click.option("--experiment_name", default=None)(f)
    f = click.option("--experiment_id", default=None)(f)
    f = click.option("--seed", default=0)(f)
    f = click.option("--verbose", is_flag=True)(f)
    return f


def training_options(f: Callable) -> Callable:
    f = click.option("--model", default="bert-base-uncased")(f)
    f = click.option("--model_type", default="Bert")(f)
    f = click.option(
        "--problem_type",
        default="single_label_classification",
        type=click.Choice(
            ["multi_label_classification", "single_label_classification"]
        ),
    )(f)
    f = click.option("--data", default="data/cookie_banner_data.csv")(f)
    f = click.option("--num_epochs", default=7)(f)
    f = click.option("--learning_rate", default=5e-5)(f)
    f = click.option("--weight_decay", default=0.0)(f)
    f = click.option("--num_epochs_2", default=10)(f)
    f = click.option("--learning_rate_2", default=5e-5)(f)
    f = click.option("--weight_decay_2", default=0.0)(f)
    f = click.option("--num_folds", default=5)(f)
    f = click.option("--no_save", is_flag=True)(f)
    f = click.option("--eval", is_flag=True)(f)
    f = click.option("--use_loss_weights", is_flag=True)(f)
    f = click.option("--two_step_model", is_flag=True)(f)
    f = click.option(
        "--task",
        type=click.Choice(
            [
                "purpose_classification",
                "purpose_classification_fs",
                "purpose_detection",
                "ie_text_classification",
                "ie_text_filtering",
                "ie_text_ps_classification",
            ]
        ),
        default="multi_class_classification",
    )(f)
    return f


def prediction_options(f: Callable) -> Callable:
    f = click.option(
        "--detection_model",
        default="models/purpose_detection_model",
    )(f)
    f = click.option(
        "--cookieblock_model", default="models/cookie_classification_model.xgb"
    )(f)
    return f


def get_args(config_file: str, **kwargs: Any) -> Namespace:
    config = load_yaml(config_file)
    for key in kwargs:
        if key not in config or (key in config and kwargs[key] is not None):
            config[key] = kwargs[key]

    args = Namespace(**config)
    args.tokenizer = args.model if args.tokenizer is None else args.tokenizer
    args.project_name = (
        args.model if args.experiment_name is None else args.experiment_name
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.experiment_name = f"{args.project_name}_{timestamp}"
    return args
