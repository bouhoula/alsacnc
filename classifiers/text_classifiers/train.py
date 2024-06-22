import logging
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

from classifiers.text_classifiers.get_args import (
    general_options,
    get_args,
    training_options,
)
from classifiers.text_classifiers.get_datasets import (
    data_collator,
    get_datasets,
    preprocess_dataset_for_ie_text_classification,
)
from classifiers.text_classifiers.metrics import get_metric_fn, get_prediction
from classifiers.text_classifiers.models import get_model
from classifiers.text_classifiers.utils import get_loss_weights, seed_everything


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        loss_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.loss_weights = loss_weights
        if self.loss_weights is not None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.loss_weights = torch.tensor(
                self.loss_weights, device=device, dtype=torch.float32
            )

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.model.config.problem_type == "single_label_classification":
            loss_fct = nn.CrossEntropyLoss(weight=self.loss_weights)
        else:
            assert self.model.config.problem_type == "multi_label_classification"
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.loss_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(
    args: Namespace,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_set, val_set, num_labels, train_labels = get_datasets(
        args, tokenizer, train_df, val_df
    )
    assert num_labels is not None
    assert train_labels is not None
    model = get_model(args.model, num_labels, args.model_type, args.problem_type)
    compute_metrics = get_metric_fn(
        args.problem_type == "multi_label_classification", args.task
    )

    train_args = TrainingArguments(
        output_dir=args.experiment_name,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="no",
        report_to="none",
        logging_strategy=IntervalStrategy.NO,
    )

    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        loss_weights=get_loss_weights(
            train_labels, args.problem_type, args.use_loss_weights
        ),
    )

    if not args.eval:
        trainer.train()
        if not args.no_save:
            model.save_pretrained(args.experiment_name)

    eval_metrics = trainer.evaluate()
    return eval_metrics


def train_2step_iect(
    args: Namespace,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    print("Training a 2-step model ..")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if train_df is None:
        assert val_df is None
        df = pd.read_csv(args.data)
        train_df, val_df = train_test_split(
            df, stratify=df["label"].apply(str), test_size=0.2
        )

    detection_experiment_name = args.experiment_name + "_detection"
    (
        detection_train_df,
        detection_labels2id,
    ) = preprocess_dataset_for_ie_text_classification(
        "ie_text_filtering", "single_label_classification", df=train_df.copy()
    )
    assert detection_labels2id["Other"] == 0
    detection_train_set = get_datasets(args, tokenizer, detection_train_df)[0]
    detection_model = get_model(
        args.model, 2, args.model_type, "single_label_classification"
    )
    detection_compute_metrics = get_metric_fn(
        do_multi_label_classification=False, task="ie_text_filtering"
    )
    detection_train_args = TrainingArguments(
        output_dir=detection_experiment_name,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="no",
        report_to="none",
    )

    detection_trainer = CustomTrainer(
        model=detection_model,
        args=detection_train_args,
        train_dataset=detection_train_set,
        compute_metrics=detection_compute_metrics,
        loss_weights=get_loss_weights(
            list(detection_train_df["label"]),
            "single_label_classification",
            args.use_loss_weights,
        ),
    )

    classification_experiment_name = args.experiment_name + "_classification"
    (
        classification_train_df,
        classification_labels2id,
    ) = preprocess_dataset_for_ie_text_classification(
        "ie_text_ps_classification", args.problem_type, df=train_df.copy()
    )
    classification_train_set, _, num_labels, _ = get_datasets(
        args, tokenizer, classification_train_df
    )
    assert num_labels is not None
    classification_model = get_model(
        args.model, num_labels, args.model_type, args.problem_type
    )
    classification_compute_metrics = get_metric_fn(
        args.problem_type == "multi_label_classification", args.task
    )
    classification_train_args = TrainingArguments(
        output_dir=classification_experiment_name,
        num_train_epochs=args.num_epochs_2,
        learning_rate=args.learning_rate_2,
        weight_decay=args.weight_decay_2,
        save_strategy="no",
        report_to="none",
    )

    classification_trainer = CustomTrainer(
        model=classification_model,
        args=classification_train_args,
        train_dataset=classification_train_set,
        compute_metrics=classification_compute_metrics,
        loss_weights=get_loss_weights(
            list(classification_train_df["label"]),
            args.problem_type,
            args.use_loss_weights,
        ),
    )

    if not args.eval:
        detection_trainer.train()
        classification_trainer.train()
        if not args.no_save:
            detection_model.save_pretrained(detection_experiment_name)
            classification_model.save_pretrained(classification_experiment_name)

    val_set = get_datasets(args, tokenizer, val_df)[0]
    val_loader = DataLoader(val_set, batch_size=8, collate_fn=data_collator)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            detection_pred = get_prediction(
                detection_model(input_ids, attention_mask).logits.detach().cpu()
            )
            classification_pred = get_prediction(
                classification_model(input_ids, attention_mask).logits.detach().cpu()
            )
            y_pred.extend(
                [
                    len(classification_labels2id) if not d_pred else c_pred
                    for (d_pred, c_pred) in zip(detection_pred, classification_pred)
                ]
            )
            y_true.extend(
                [
                    len(classification_labels2id)
                    if label == "Other"
                    else classification_labels2id[label]
                    for label in batch["label"]
                ]
            )

    eval_accuracy = accuracy_score(y_true, y_pred)
    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    eval_metrics = {
        "eval_accuracy": eval_accuracy,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
    }
    print(eval_metrics)
    print(confusion_matrix(y_true, y_pred))
    return eval_metrics


@click.command()
@general_options
@training_options
def main(config_file: str, seed: int, **kwargs: Any) -> None:
    seed_everything(seed)
    args = get_args(config_file, **kwargs)
    print(args)
    if args.two_step_model:
        train_2step_iect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
