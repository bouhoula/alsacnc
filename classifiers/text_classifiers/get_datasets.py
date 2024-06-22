import json
import logging
from argparse import Namespace
from typing import Any, Dict, Generator, List, Optional, Tuple

import datasets
import pandas as pd
import torch
from datasets import Dataset
from datasets.arrow_dataset import Batch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from database.queries import get_cb_text_with_predictions, get_table
from shared_utils import read_txt_file

logging.basicConfig(level=logging.ERROR)


def preprocess_dataset_for_purpose_classification(args: Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.data)
    if args.task == "purpose_classification":
        df = df[df["Code"].str.contains("|".join(args.raw_labels))]
        num_clusters = len(set(args.label_clusters))
        cluster_names = []
        cluster_sizes = []
        for cluster_id in range(num_clusters):
            cluster_labels = [
                l
                for i, l in enumerate(args.raw_labels)
                if args.label_clusters[i] == cluster_id
            ]
            cluster_names.append(f"cluster_{cluster_id}")
            df[cluster_names[-1]] = (
                df[args.label_column].str.contains("|".join(cluster_labels)).astype(int)
            )
            cluster_sizes.append(df[cluster_names[-1]].sum())
        df = df.groupby(args.group_by_columns).sum().reset_index()
        for cluster_name in cluster_names:
            df[cluster_name] = df[cluster_name].apply(lambda x: min(x, 1))
        df["text"] = df[args.text_column]
        df["label"] = df[cluster_names].values.tolist()
    else:
        assert args.task == "purpose_classification_fs"
        labels: List = [[] for _ in set(args.label_clusters)]
        for label_cluster_num, label in zip(args.label_clusters, args.raw_labels):
            labels[label_cluster_num].append(label)
        df["label"] = df["label"].apply(
            lambda x: [
                int(any([y in label_cluster for y in json.loads(x.replace("'", '"'))]))
                for label_cluster in labels
            ]
        )
        cluster_sizes = [
            df["label"].apply(lambda x: x[i]).sum() for i in range(len(labels))
        ]
        counts: Dict[str, int] = {}
        for label in df["label"]:
            counts[str(label)] = 1 + counts.get(str(label), 0)
        print(counts)
    df["labels"] = df["label"].apply(lambda x: "".join(map(str, x)))
    df["num_labels"] = df["label"].apply(lambda x: sum(x))
    if args.problem_type == "multi_label_classification":
        df["id_1"] = df["label"].apply(lambda x: "_".join(map(str, x)))
        df["id_2"] = df["label"].apply(
            lambda x: max(
                enumerate(zip(x, cluster_sizes)), key=lambda y: (y[1][0], -y[1][1])
            )[0]
        )
        df = df[["text", "label", "id_1", "id_2", "labels"]]
    else:
        num_labels = len(df.iloc[0]["label"])
        if num_labels == 2:
            df["label"] = df["label"].apply(lambda x: int(x[1] > 0))
            print((df["label"] == 0).sum(), (df["label"] == 1).sum())
            # print(list(df[df["label"] == 0]["text"]))
        else:
            df = df[df["num_labels"] == 1]
            df.loc[:, "label"] = df["label"].apply(lambda x: x.index(1))
        df = df[["text", "label", "labels"]]
    return df


def preprocess_dataset_for_ie_text_classification(
    task: str,
    problem_type: str,
    data_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df is None:
        assert data_path is not None
        df = pd.read_csv(data_path)
    if task == "ie_text_ps_classification":
        df = df[df["label"] != "Other"]
    elif task == "ie_text_filtering":
        df.loc[:, "label"] = df["label"].apply(
            lambda x: "Positive sample" if x != "Other" else "Other"
        )
    labels = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    df = df.reset_index(drop=True)
    df.loc[:, "label"] = df["label"].apply(lambda x: labels[x])
    if problem_type == "multi_label_classification":
        df.loc[:, "label"] = df["label"].apply(
            lambda x: [int(x == i) for i in range(len(labels))]
        )
    return df, labels


def get_data_for_cross_validation(args: Namespace) -> Tuple[pd.DataFrame, Generator]:
    kf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    if args.task in ["purpose_classification", "purpose_classification_fs"]:
        df = preprocess_dataset_for_purpose_classification(args)
        if args.problem_type == "multi_label_classification":
            generator = kf.split(df, df["id_2"])
            df = df.drop(columns=["id_1", "id_2"])
        else:
            generator = kf.split(df, df["label"])
    elif args.task in [
        "ie_text_classification",
        "ie_text_ps_classification",
        "ie_text_filtering",
    ]:
        df = pd.read_csv(args.data)
        if not args.two_step_model:
            df, _ = preprocess_dataset_for_ie_text_classification(
                args.task,
                args.problem_type,
                df=df,
            )
        generator = kf.split(df, df["label"].apply(str))
    elif args.task == "purpose_detection":
        df = pd.read_csv(args.data)
        generator = kf.split(df, df["label"])
    else:
        raise ValueError(f"Unrecognized task {args.task}")
    return df, generator


def get_train_val_dfs(args: Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.task in ["purpose_classification", "purpose_classification_fs"]:
        df = preprocess_dataset_for_purpose_classification(args)
    elif args.task in [
        "ie_text_classification",
        "ie_text_filtering",
        "ie_text_ps_classification",
    ]:
        df, _ = preprocess_dataset_for_ie_text_classification(
            args.task, args.problem_type, data_path=args.data
        )
    elif args.task == "purpose_detection":
        df = pd.read_csv(args.data)
    else:
        raise ValueError(f"Unrecognized task {args.task}")

    if "id_1" in df.columns:
        try:
            print("Splitting with id_1")
            train_df, val_df = train_test_split(df, stratify=df["id_1"], test_size=0.2)
        except ValueError:
            print("Splitting with id_2")
            train_df, val_df = train_test_split(df, stratify=df["id_2"], test_size=0.2)
        train_df = train_df.drop(columns=["id_1", "id_2"])
        val_df = val_df.drop(columns=["id_1", "id_2"])
    else:
        train_df, val_df = train_test_split(
            df, stratify=df["label"].apply(str), test_size=0.2
        )
    return train_df, val_df


def get_num_labels(df: pd.DataFrame) -> int:
    sample = df.iloc[0]["label"]
    if isinstance(sample, list):
        return len(sample)
    else:
        return len(df["label"].unique())


def get_datasets(
    args: Namespace,
    tokenizer: PreTrainedTokenizer,
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dataset, Optional[Dataset], Optional[int], Optional[List]]:
    if train_df is None:
        assert val_df is None
        train_df, val_df = get_train_val_dfs(args)

    def tokenize(x: Batch) -> BatchEncoding:
        return tokenizer(x["text"], **args.tokenizer_args)

    ignored_columns = []
    if "__index_level_0__" in train_df.columns:
        ignored_columns.append("__index_level_0__")

    datasets.logging.set_verbosity_error()
    train_set = Dataset.from_pandas(train_df)
    tokenized_train_set = train_set.map(
        tokenize, batched=True, batch_size=20, remove_columns=ignored_columns
    )

    tokenized_val_set = None
    if val_df is not None:
        val_set = Dataset.from_pandas(val_df)
        tokenized_val_set = val_set.map(
            tokenize, batched=True, batch_size=20, remove_columns=ignored_columns
        )
    num_labels, labels = None, None
    try:
        num_labels = get_num_labels(train_df)
        labels = list(train_df["label"])
    except:
        pass
    return (
        tokenized_train_set,
        tokenized_val_set,
        num_labels,
        labels,
    )


def get_dataframe_from_text_files(
    purpose_text_file: str,
    non_purpose_text_file: str,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    purpose_text = list(set(read_txt_file(purpose_text_file)))
    non_purpose_text = list(set(read_txt_file(non_purpose_text_file)))
    text = purpose_text + non_purpose_text
    label = [1] * len(purpose_text) + [0] * len(non_purpose_text)
    df = pd.DataFrame(list(zip(text, label)), columns=["text", "label"])
    if output_file is not None:
        df.to_csv(output_file, index=False)
    return df


def data_collator(features: List[Any]) -> Dict[str, Any]:
    first = features[0]
    batch = {}

    for k, v in first.items():
        if v is None:
            continue
        if isinstance(v, str):
            batch[k] = [f[k] for f in features]
        else:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def get_dataloader_from_db(
    args: Namespace,
    tokenizer: PreTrainedTokenizer,
    prediction_table_name: Optional[str] = None,
    extraction_mode: str = "all",
    website_ids: Optional[List[int]] = None,
) -> DataLoader:
    if extraction_mode == "all":
        filter_dict = (
            dict(website_id=(1, website_ids)) if website_ids is not None else None
        )
        df = get_table("cb_text", filter=filter_dict)
    else:
        assert prediction_table_name is not None
        df = get_cb_text_with_predictions(prediction_table_name, extraction_mode)
    needed_columns = ["id", "text", "website_id", "num_clicks"]
    if "id_1" in df.columns:
        needed_columns.append("id_1")
    df = df[needed_columns]
    dataset = get_datasets(args, tokenizer, df)[0]
    return DataLoader(dataset, batch_size=16, collate_fn=data_collator)
