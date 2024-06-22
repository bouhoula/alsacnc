from argparse import Namespace
from datetime import datetime
from typing import Dict

import click
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification

from classifiers.text_classifiers.get_args import (
    general_options,
    get_args,
    prediction_options,
)
from classifiers.text_classifiers.get_datasets import get_dataloader_from_db
from classifiers.text_classifiers.metrics import get_prediction
from database.queries import (
    get_entry,
    get_last_experiment,
    get_table,
    init_db,
    insert_into_db,
    update_entry,
)


def compute_expiry_time_in_seconds(
    start_ts: pd.Timestamp, end_ts: datetime, session: int
) -> int:
    if session:
        return 0
    else:
        if isinstance(end_ts, str):
            end_ts = datetime.fromisoformat(end_ts[:-1])
        timedelta = end_ts - start_ts.to_pydatetime()
        return int(timedelta.total_seconds())


class Predictor:
    def __init__(self, args: Namespace):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Device:", self.device)
        self.models = {
            "purpose_detection": BertForSequenceClassification.from_pretrained(
                args.detection_model, num_labels=2
            ),
            "cookieblock": args.cookieblock_model,
        }
        for _, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
        self.verbose = args.verbose

    def make_purpose_predictions(
        self,
        model_name: str,
        prediction_table_name: str,
        experiment_id: str,
    ) -> None:
        website_ids = get_table(
            "websites", filter=dict(experiment_id=(1, experiment_id))
        )["id"].tolist()
        dataloader = get_dataloader_from_db(
            self.args,
            self.tokenizer,
            prediction_table_name,
            extraction_mode="all",
            website_ids=website_ids,
        )
        if self.verbose:
            print("Dataloader created ..")
        counts = {}
        records = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentence_ids = batch["id"].detach().cpu().tolist()
                logits = self.models[model_name](input_ids, attention_mask).logits
                websites_ids = batch["website_id"].tolist()
                num_clicks_list = batch["num_clicks"].tolist()
                predictions = get_prediction(logits.detach().cpu().numpy()).tolist()
                records.extend(
                    [
                        dict(sentence_id=sentence_id, purpose_detected=prediction)
                        for sentence_id, prediction in zip(sentence_ids, predictions)
                    ]
                )
                for prediction, website_id, num_clicks in zip(
                    predictions, websites_ids, num_clicks_list
                ):
                    if website_id not in counts:
                        counts[website_id] = [0, 0]
                    counts[website_id][min(num_clicks, 1)] += prediction
        for website_id in counts:
            crawl_results = get_entry(
                "crawl_results", filter=dict(website_id=(1, website_id))
            )
            if crawl_results is None:
                print(f"Warning: Crawl results is None for website_id {website_id}.")
                continue
            crawl_results["tracking_purposes_detected_in_initial_text"] = int(
                counts[website_id][0] > 0
            )
            crawl_results["tracking_purposes_detected"] = int(
                (counts[website_id][0] + counts[website_id][1]) > 0
            )
            update_entry(crawl_results)
        insert_into_db(prediction_table_name, records)

    def predict(self, experiment_id: str) -> None:
        self.make_purpose_predictions(
            "purpose_detection", "purpose_predictions", experiment_id
        )


@click.command()
@general_options
@prediction_options
def main(config_file: str, **kwargs: Dict) -> None:
    init_db("postgres", create_tables=True)
    args = get_args(config_file, **kwargs)
    predictor = Predictor(args)

    experiment_ids = (
        [get_last_experiment()] if args.experiment_id is None else [args.experiment_id]
    )

    for experiment_id in experiment_ids:
        print("Starting prediction for", experiment_id)
        predictor.predict(experiment_id)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
