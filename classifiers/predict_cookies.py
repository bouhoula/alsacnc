from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd

from classifiers.cookieblock_classifier.predict import predict as cookieblock_predict
from classifiers.text_classifiers.get_args import (
    general_options,
    get_args,
    prediction_options,
)
from database.queries import (
    create_engine_for_openwpm_db,
    get_entry,
    get_last_experiment,
    get_table,
    init_db,
    insert_into_db,
    update_entry,
    update_missing_entries_in_crawl_results,
)

NUM_POOL = min(cpu_count() - 1, 16)
COOKIEBLOCK_MODEL_PATH: Optional[str] = None


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


def make_cookie_predictions_iteration(
    website_and_cookies: Tuple[Dict, pd.DataFrame]
) -> None:
    website, cookies_df = website_and_cookies
    crawl_results = get_entry(
        "crawl_results", filter=dict(website_id=(1, website["id"]))
    )
    if crawl_results is None:
        return

    cookies_df = cookies_df[cookies_df["record_type"] != "deleted"]
    cookies_dict = {}
    for _, row in cookies_df.iterrows():
        cookie_key = (
            f"{row['name']};{row['cookie_domain']};{row['path']};"
            f"{row['website_id']};{row['collection_strategy']}"
        )
        if cookie_key not in cookies_dict:
            cookies_dict[cookie_key] = {
                "visit_id": row["visit_id"],
                "name": row["name"],
                "cookie_domain": row["cookie_domain"],
                "path": row["path"],
                "website": website["name"],
                "website_id": website["id"],
                "timestamp": row["timestamp"],
                "collection_strategy": row["collection_strategy"],
                "variable_data": [],
            }

        cookies_dict[cookie_key]["variable_data"].append(
            {
                "value": row["value"],
                "expiry": compute_expiry_time_in_seconds(
                    row["timestamp"], row["expiry"], int(row["is_session"])
                ),
                "session": bool(row["is_session"]),
                "http_only": bool(row["is_http_only"]),
                "host_only": bool(row["is_host_only"]),
                "secure": bool(row["is_secure"]),
                "same_site": row["same_site"],
            }
        )

    if len(cookies_dict) == 0:
        return
    assert isinstance(COOKIEBLOCK_MODEL_PATH, str)
    predictions = cookieblock_predict(cookies_dict, COOKIEBLOCK_MODEL_PATH)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for cookie_key in cookies_dict:
        cookie = cookies_dict[cookie_key]
        records.append(
            {
                key: cookie[key]
                for key in [
                    "visit_id",
                    "name",
                    "cookie_domain",
                    "path",
                    "website_id",
                    "timestamp",
                    "collection_strategy",
                ]
            }
        )
        records[-1]["classification"] = predictions[cookie_key]
        if predictions[cookie_key] == 1:
            collection_strategy = records[-1]["collection_strategy"]
            counts[collection_strategy] = 1 + counts.get(collection_strategy, 0)
    insert_into_db("cookies_with_predictions", records)
    assert isinstance(crawl_results, dict)
    crawl_results["tracking_detected"] = max(
        counts.get("No interaction", 0), counts.get("Accept", 0)
    )
    if crawl_results["cookie_notice_detected"]:
        crawl_results["tracking_detected_prior_to_interaction"] = counts.get(
            "No interaction", 0
        )
        if "Reject" in counts:
            crawl_results["tracking_detected_after_reject"] = counts["Reject"]
        if "Close/Continue without accepting" in counts:
            crawl_results["tracking_detected_after_close"] = counts[
                "Close/Continue without accepting"
            ]
        if "Save cookie settings" in counts:
            crawl_results["tracking_detected_after_save"] = counts[
                "Save cookie settings"
            ]
    update_entry(crawl_results)


def copy_cookies_iteration(ts: Dict, experiment_id: str) -> Dict:
    start_timestamp = ts["start_timestamp"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_timestamp = ts["end_timestamp"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    openwpm_engine = create_engine_for_openwpm_db(experiment_id)
    cookies = pd.read_sql(
        "SELECT * FROM javascript_cookies WHERE visit_id = ? AND ? <= time_stamp AND time_stamp <= ?",
        openwpm_engine,
        params=(str(ts["visit_id"]), start_timestamp, end_timestamp),
    )
    assert cookies["time_stamp"].apply(lambda x: x[-1] == "Z").all()
    cookies["timestamp"] = cookies["time_stamp"].apply(
        lambda x: datetime.fromisoformat(x[:-1])
    )
    cookies["cookie_domain"] = cookies["host"]
    cookies.drop(["time_stamp", "host", "first_party_domain"], axis=1, inplace=True)
    cookies["collection_strategy"] = ts["collection_strategy"]
    cookies["website_id"] = ts["website_id"]
    cookies["expiry"] = cookies["expiry"].apply(
        lambda x: x if not x.startswith("+") else "9999-12-31T21:59:59.000Z"
    )
    if "id" in cookies:
        del cookies["id"]
    return cookies.to_dict("records")


class CookieCopier:
    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id

    def __call__(self, ts_row: Dict) -> Dict:
        _, ts = ts_row
        return copy_cookies_iteration(ts, self.experiment_id)


def make_cookie_predictions(
    experiment_id: str, batch_size: int = 1000, offset: int = 0
) -> None:
    experiment_id_filter = dict(experiment_id=(1, experiment_id))
    websites_ids = get_table("websites", filter=experiment_id_filter)["id"].tolist()

    while True:
        if offset >= len(websites_ids):
            break

        filter_dict = dict(id=(1, websites_ids[offset : offset + batch_size]))
        filter_dict.update(experiment_id_filter)

        cookie_timestamps = get_table(
            "websites",
            mode="join",
            subquery_table="cookie_timestamps",
            subquery_col="website_id",
            filter=filter_dict,
        )

        with Pool(NUM_POOL) as pool:
            cookies = pool.map(
                CookieCopier(experiment_id), cookie_timestamps.iterrows()
            )
            cookies = [record for result in cookies for record in result]
            print(
                f"Extracted batch {1+offset//batch_size} cookies for experiment {experiment_id}"
            )
            cookies_dfs = {
                k: v for k, v in list(pd.DataFrame(cookies).groupby("website_id"))
            }
            websites = get_table(
                "websites",
                filter=dict(id=(1, list(cookies_dfs.keys()))),
                return_df=False,
            )
            websites_and_cookies = [
                (website, cookies_dfs[website["id"]]) for website in websites
            ]
            pool.map(make_cookie_predictions_iteration, websites_and_cookies)
            print(f"Completed predictions for batch {1+offset//batch_size}")

        offset += batch_size


@click.command()
@general_options
@prediction_options
def main(config_file: str, **kwargs: Dict) -> None:
    init_db("postgres", create_tables=True)
    args = get_args(config_file, **kwargs)
    global COOKIEBLOCK_MODEL_PATH
    COOKIEBLOCK_MODEL_PATH = args.cookieblock_model

    experiment_ids = (
        [get_last_experiment()] if args.experiment_id is None else [args.experiment_id]
    )

    for experiment_id in experiment_ids:
        make_cookie_predictions(experiment_id)

    update_missing_entries_in_crawl_results()


if __name__ == "__main__":
    main()
