from typing import Any, Dict, List

import click
import numpy as np

from classifiers.text_classifiers.get_args import (
    general_options,
    get_args,
    training_options,
)
from classifiers.text_classifiers.get_datasets import get_data_for_cross_validation
from classifiers.text_classifiers.train import train, train_2step_iect
from classifiers.text_classifiers.utils import seed_everything


def run_kfold(
    config_file: str, returned_metric: str = "accuracy", **kwargs: Any
) -> float:
    kwargs["no_save"] = True
    args = get_args(config_file, **kwargs)
    seed_everything(args.seed)
    metrics: Dict[str, List[float]] = {
        "eval_accuracy": [],
        "eval_precision": [],
        "eval_recall": [],
        "eval_f1": [],
    }
    df, split_generator = get_data_for_cross_validation(args)
    for i, (train_idx, val_idx) in enumerate(split_generator):
        print(f"Starting fold {i+1}")
        train_fn = train_2step_iect if args.two_step_model else train
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        tmp_metrics = train_fn(args, train_df, val_df)
        for key in metrics:
            metrics[key].append(tmp_metrics[key])

    print("\n\nFinished running cross-validation. Metrics:")
    for key in metrics:
        print(f"{key.replace('eval_', '')}: {np.mean(metrics[key])}")
    print("\n")

    return np.mean(metrics[f"eval_{returned_metric}"]).item()


@click.command()
@general_options
@training_options
def main(config_file: str, **kwargs: Any) -> None:
    run_kfold(config_file, **kwargs)


if __name__ == "__main__":
    main()
