import io
import warnings
from contextlib import redirect_stdout
from typing import Any, Dict

import click
import datasets
import optuna

from classifiers.text_classifiers.get_args import general_options, training_options
from classifiers.text_classifiers.kfold import run_kfold


def hp_space(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [5e-6, 1e-6, 5e-5]),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 1e-6, 5e-6, 1e-5]
        ),
        "num_epochs": trial.suggest_int("num_epochs", 5, 20, step=5),
    }


HP_SPACE = {
    "learning_rate": [5e-6, 7e-6, 1e-5],
    "weight_decay": [0, 1e-6, 5e-6, 1e-5],
    "num_epochs": [10, 15, 20, 25],
}


def objective(
    config_file: str, metric: str, trial: optuna.Trial, **kwargs: Any
) -> float:
    trial_params = hp_space(trial)
    for key in trial_params:
        kwargs[key] = trial_params[key]
    with redirect_stdout(io.StringIO()):
        result = run_kfold(config_file, returned_metric=metric, **kwargs)
    return result


@click.command()
@general_options
@training_options
@click.option(
    "--metric",
    type=click.Choice(["accuracy", "f1", "recall", "precision"]),
    default="accuracy",
)
@click.option("--n_trials", default=1000)
def optimize_hp(config_file: str, metric: str, n_trials: int, **kwargs: Any) -> None:
    datasets.utils.logging.disable_progress_bar()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    experiment_name = kwargs["experiment_name"]
    study = optuna.create_study(study_name=experiment_name, direction="maximize")
    study.optimize(
        lambda trial: objective(config_file, metric, trial, **kwargs), n_trials=n_trials
    )


if __name__ == "__main__":
    optimize_hp()
