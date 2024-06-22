import inspect
import os
import shutil
import sys
from ast import literal_eval
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import pandas as pd
from sqlalchemy import create_engine, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, subqueryload
from sqlalchemy.pool import NullPool

import database.tables as tables
from shared_utils import read_txt_file, repeat

Session = sessionmaker(expire_on_commit=False)
OpenWPMSession = sessionmaker()

table_classes = [
    cls
    for name, cls in inspect.getmembers(sys.modules["database.tables"])
    if inspect.isclass(cls) and issubclass(cls, tables.Base) and cls != tables.Base
]
map_table_name_to_class = {cls.__tablename__: cls for cls in table_classes}


@repeat()
def init_db(
    engine_name: str, create_tables: bool = True, drop_existing_tables: bool = False
) -> None:
    if engine_name == "postgres":
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    elif engine_name == "sqlite":
        db_url = f"sqlite:///{os.getenv('DB_PATH')}"
    else:
        raise ValueError(f"Unrecognized engine name {engine_name}")
    engine = create_engine(db_url, poolclass=NullPool)
    Session.configure(bind=engine)
    if drop_existing_tables:
        with Session() as session:
            for table in reversed(tables.Base.metadata.sorted_tables):
                session.execute(f"DROP TABLE {table.name} CASCADE")
            session.commit()
        print("Dropped existing tables.")
    if create_tables:
        tables.Base.metadata.create_all(engine)
    print(f"Connected to {engine}")


@repeat()
def create_entry(entry_dict: Dict) -> None:
    with Session() as session:
        entry = (
            session.query(entry_dict["__tablename__"])
            .filter_by(id=entry_dict["id"])
            .one()
        )
        entry.update(**entry_dict)
        session.add(entry)
        session.commit()


@repeat()
def update_entry(entry_dict: Dict) -> None:
    with Session() as session:
        entry = (
            session.query(entry_dict["__tablename__"])
            .filter_by(id=entry_dict["id"])
            .one()
        )
        entry.update(**entry_dict)
        session.add(entry)
        session.commit()


@repeat()
def delete_entry(entry_dict: Dict) -> None:
    with Session() as session:
        entry = (
            session.query(entry_dict["__tablename__"])
            .filter_by(id=entry_dict["id"])
            .one()
        )
        session.delete(entry)
        session.commit()


@repeat()
def clear_table(table: Union[str, tables.Base]) -> None:
    if isinstance(table, str):
        table = map_table_name_to_class[table]
    with Session() as session:
        session.query(table).delete()
        session.commit()


@repeat(sleep=10)
# flake8: noqa: C901
def get_table(
    table: Union[str, tables.Base],
    session_cls: sessionmaker = Session,
    mode: str = "normal",
    subquery_col: Any = None,
    subquery_table: Any = None,
    filter: Any = None,
    filter_expr: Any = None,
    subquery_cols: Optional[List[str]] = None,
    return_df: bool = True,
) -> Union[pd.DataFrame, List[tables.Base]]:
    if isinstance(table, str):
        table = map_table_name_to_class[table]
    with session_cls() as session:
        query = session.query(table)
        if subquery_cols is not None:
            cols = []
            for s in subquery_cols:
                cols.append(reduce(lambda x, y: getattr(x, y), s.split("."), table))
            if len(cols) > 0:
                query = query.options(*[subqueryload(col) for col in cols])
        if mode == "exclude":
            assert subquery_col is not None
            subquery = session.query(subquery_col)
            query = query.filter(table.id.not_in(subquery))
        elif mode == "join":
            assert subquery_table is not None and subquery_col is not None
            if isinstance(subquery_table, str):
                subquery_table = map_table_name_to_class[subquery_table]
            if isinstance(subquery_col, str):
                subquery_col = reduce(
                    lambda x, y: getattr(x, y), subquery_col.split("."), subquery_table
                )
            query = session.query(table, subquery_table)
            query = query.join(subquery_table, subquery_col == table.id)

        if filter is not None:
            for key, (eq, value) in filter.items():
                try:
                    entry = reduce(lambda x, y: getattr(x, y), key.split("."), table)
                except Exception as e:
                    print(f"Warning: could not find column {table}.{key}")
                    raise e
                    continue

                if value is None:
                    if eq:
                        condition = entry.is_(None)
                    else:
                        condition = entry.isnot(None)
                elif isinstance(value, List):
                    if eq:
                        condition = entry.in_(value)
                    else:
                        condition = entry.not_in(value)
                else:
                    if eq:
                        condition = entry == value
                    else:
                        condition = entry != value

                query = query.filter(condition)
        if filter_expr is not None:
            query = query.filter(filter_expr)

    if return_df:
        df = pd.read_sql(query.statement, query.session.bind)
        return df

    entries = []
    for entry in query:
        entries.append(entry.to_dict())
    return entries


def get_and_postprocess_table(
    table_name: str,
    filter: Dict,
    supported_languages: Optional[Tuple[str, ...]] = None,
    include_iab_tcf: bool = False,
    include_extensions: bool = False,
) -> pd.DataFrame:
    if table_name == "websites":
        table = get_table(
            "websites",
            filter=filter,
        )
    else:
        table = get_table(
            "websites",
            mode="join",
            subquery_table=table_name,
            subquery_col="website_id",
            filter=filter,
        )

    if supported_languages is not None:
        table = table[table["language"].isin(supported_languages)].reset_index()
    cmps = read_txt_file("config/cmp_list.txt")
    table["consentomatic"] = table["cmp"].apply(
        lambda x: [] if x is None else list(set(x["consentomatic"]).intersection(cmps))
    )
    table["cmp_summary"] = table["consentomatic"].apply(
        lambda x: "No CMP" if x is None or len(x) == 0 else x[0]
    )
    if include_iab_tcf:
        table["uses_tcf_iab"] = table["cmp"].apply(
            lambda x: x is not None
            and "tcfapi" in x
            and x["tcfapi"] is not None
            and "cmpId" in x["tcfapi"]
        )
    if include_extensions:
        table["extension"] = table["name"].apply(lambda x: x.split(".")[-1])
    crux_data = table["ranking_data"].iloc[0]["crux"]
    if isinstance(crux_data, str):
        crux_data = literal_eval(crux_data)
    countries = list(crux_data.keys())
    for country in countries:
        if isinstance(table["ranking_data"].iloc[0]["crux"], str):
            table[country] = table["ranking_data"].apply(
                lambda x: literal_eval(x["crux"])[country]
            )
        else:
            table[country] = table["ranking_data"].apply(lambda x: x["crux"][country])
    return table


def get_entry(
    table: Union[str, tables.Base],
    filter: Any = None,
    subquery_cols: Optional[List[str]] = None,
) -> Optional[Dict]:
    entries = get_table(
        table, filter=filter, subquery_cols=subquery_cols, return_df=False
    )
    if len(entries) == 0:
        return None
    else:
        if len(entries) > 1:
            print(
                f"Warning found {len(entries)} entries when querying table {table}. Expected 1."
            )
        return entries[0]


def get_cb_text_with_predictions(
    prediction_table_name: str,
    extraction_mode: str = "all",
    filter: Optional[Dict] = None,
) -> pd.DataFrame:
    assert extraction_mode in ["all", "exclude_predicted", "include_predicted_only"]
    pred_table = map_table_name_to_class[prediction_table_name]
    query_mode, subquery_table, subquery_col, filter_expr = (
        None,
        None,
        None,
        None,
    )
    if extraction_mode == "exclude_predicted":
        subquery_col = pred_table.sentence_id
        query_mode = "exclude"
    elif extraction_mode in ["all", "include_predicted_only"]:
        if extraction_mode == "include_predicted_only":
            filter_expr = (pred_table.purpose_detected == 1) & (
                pred_table.purpose_classification.is_(None)
            )
        subquery_table = pred_table
        subquery_col = pred_table.sentence_id
        query_mode = "join"
    else:
        raise ValueError(f"Unrecognized extraction mode {extraction_mode}")
    return get_table(
        "cb_text",
        mode=query_mode,
        subquery_table=subquery_table,
        subquery_col=subquery_col,
        filter=filter,
        filter_expr=filter_expr,
    )


@repeat(sleep=10)
def insert_into_db(
    table_name: str, records: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[Dict, List[Dict]]:
    table = map_table_name_to_class[table_name]
    if isinstance(records, dict):
        records = [records]
    entries = []
    with Session() as session:
        for record in records:
            entry = table(**record)
            session.add(entry)
            entries.append(entry)
        session.commit()
        entries = [entry.to_dict() for entry in entries]
    return entries[0] if len(records) == 1 else entries


def update_db(
    table_name: str,
    filter_keys: Union[Tuple[str, ...], str],
    filter_values_list: Union[List, Any],
    update_dicts: Union[List[Dict], Dict],
) -> None:
    table = map_table_name_to_class[table_name]
    if isinstance(filter_keys, str):
        filter_keys = (filter_keys,)
    if not isinstance(filter_values_list, List):
        filter_values_list = [filter_values_list]
    if isinstance(update_dicts, dict):
        update_dicts = [update_dicts]
    with Session() as session:
        for filter_values, update_dict in zip(filter_values_list, update_dicts):
            query = session.query(table)
            if not isinstance(filter_values, tuple):
                filter_values = (filter_values,)
            for filter_key, filter_value in zip(filter_keys, filter_values):
                assert hasattr(table, filter_key)
                query = query.filter(getattr(table, filter_key) == filter_value)
            query.update(update_dict)
        session.commit()


def update_db_column(
    table_name: str,
    column_name: str,
    new_value: Any,
) -> None:
    table = map_table_name_to_class[table_name]
    column = table.__table__.c[column_name]
    with Session() as session:
        session.query(table).update({column: new_value})
        session.commit()


def insert_exception_into_db(website: Dict, exception: Union[Exception, str]) -> None:
    insert_into_db(
        "errors",
        dict(
            website_id=website["id"],
            text=str(exception),
        ),
    )


@repeat(sleep=10)
def remove_from_db(website: str, experiment_id: str) -> None:
    with Session() as session:
        for table in table_classes:
            if hasattr(table, "website") and hasattr(table, "experiment_id"):
                delete_stmt = table.__table__.delete().where(
                    (table.website == website) & (table.experiment_id == experiment_id)
                )
                session.execute(delete_stmt)
        session.commit()


def query_sql(sql_command: str, params: Dict) -> pd.DataFrame:
    with Session() as session:
        result = session.execute(text(sql_command), params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df


def check_for_timeout(website: Dict) -> str:
    timeouts = get_entry(
        tables.Errors, filter=dict(website_id=(1, website["id"]), text=(1, "Timeout"))
    )
    cookie_banner = get_entry(
        tables.CookieBanner, filter=dict(website_id=(1, website["id"]))
    )
    if timeouts is None:
        return "No timeout"
    elif cookie_banner is None:
        return "Timeout during exploration"
    else:
        return "Timeout during loading"


def create_engine_for_openwpm_db(experiment_id: str) -> Engine:
    return create_engine(
        f"sqlite:///experiments/{experiment_id}/crawl.sqlite", poolclass=NullPool
    )


def export_interface_interference_screenshots(
    experiment_id: str, verbose: bool
) -> None:
    experiment_dir = Path("experiments") / experiment_id
    ii_dir = experiment_dir / "interface_interference"
    destination_dirs = [ii_dir / "not_detected", ii_dir / "detected"]
    for destination_dir in destination_dirs:
        destination_dir.mkdir(exist_ok=True, parents=True)
    crawl_results = get_table(
        "websites",
        mode="join",
        subquery_table="crawl_results",
        subquery_col="website_id",
        filter={"experiment_id": (1, experiment_id)},
    )
    crawl_results = crawl_results[
        crawl_results["interface_interference_analysis"].apply(bool)
    ]
    for _, row in crawl_results.iterrows():
        website = row["name"]
        source_file = experiment_dir / "data" / website / "cookie_banner_screenshot.png"
        ii_detected = row["interface_interference_detected"]
        if source_file.is_file():
            shutil.copy(source_file, destination_dirs[ii_detected])
            (destination_dirs[ii_detected] / "cookie_banner_screenshot.png").rename(
                f"{destination_dirs[ii_detected] / website}.png"
            )
            if verbose:
                print(f"Copied for {website} ({ii_detected})")


def list_experiments() -> None:
    experiment_ids = get_table("experiments")["id"]
    for experiment_id in experiment_ids:
        print(experiment_id)


def get_last_experiment() -> str:
    return get_table("experiments")["id"].tolist()[-1]


def show_num_websites(
    experiment_id: str, attribute: Optional[str] = None, verbose: bool = False
) -> None:
    experiment_id_filter = dict(experiment_id=(1, experiment_id))
    df = get_table("websites", filter=experiment_id_filter)
    print(f"Number of crawled websites for experiment `{experiment_id}`: {len(df)}")
    if attribute is not None:
        table_names = ["crawl_results", "cookie_banners"]
        if attribute not in df.columns:
            for table_name in table_names:
                df = get_table(
                    "websites",
                    mode="join",
                    subquery_table=table_name,
                    subquery_col="website_id",
                    filter=experiment_id_filter,
                )
                if attribute in df.columns:
                    break
        print(f"Number of websites with {attribute}: {len(df[df[attribute] > 0])}")
        if verbose:
            for website in df[df[attribute] > 0]["name"].tolist():
                print(website)


def show_errors(experiment_id: str) -> None:
    errors = get_table(
        "websites",
        mode="join",
        subquery_table="errors",
        subquery_col="website_id",
        filter=dict(experiment_id=(1, experiment_id)),
    )
    counter = 0
    for website, text in zip(errors["name"], errors["text"]):
        if "Timeout" not in text:
            continue
        print(website, ": ", text)
        counter += 1
    print(f"Errors found in {counter} websites")


def delete_websites_where_crawl_was_interrupted(experiment_id: str) -> None:
    filter = dict(success=(1, None), experiment_id=(1, experiment_id))
    with Session() as session:
        session.expire_all()
    websites = get_table("websites", filter=filter, return_df=False)
    print(f"Deleting {len(websites)} websites")
    for website in websites:
        print(website["id"], website["name"], " ...")
        delete_entry(website)
    print("Done")


def update_missing_entries_in_crawl_results() -> None:
    with Session() as session:
        for co in ["reject", "close", "save"]:
            tracking_column_name = f"tracking_detected_after_{co}"
            tracking_column = getattr(tables.CrawlResults, tracking_column_name)
            button_column = getattr(tables.CrawlResults, f"{co}_button_detected")
            session.execute(
                update(tables.CrawlResults)
                .values({tracking_column_name: 0})
                .where(button_column > 0)
                .where(tracking_column.is_(None))
            )
        session.execute(
            update(tables.CrawlResults)
            .values(dict(tracking_detected_prior_to_interaction=0))
            .where(tables.CrawlResults.cookie_notice_detected == True)
            .where(tables.CrawlResults.tracking_detected_prior_to_interaction.is_(None))
        )
        session.execute(
            update(tables.CrawlResults)
            .values(dict(tracking_detected=0))
            .where(tables.CrawlResults.tracking_detected.is_(None))
        )
        session.commit()


@click.command()
@click.option("--experiment_id", type=str, default=None)
@click.option(
    "--command",
    type=click.Choice(
        [
            "delete",
            "export_ii",
            "ls",
            "num_websites",
            "show_errors",
            "clean",
        ]
    ),
)
@click.option("--attr", type=str, default=None)
@click.option("--verbose", is_flag=True)
def main(
    experiment_id: Optional[str], command: str, attr: Optional[str], verbose: bool
) -> None:
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    init_db("postgres", create_tables=True)
    if command == "ls":
        list_experiments()
    else:
        if experiment_id is None:
            experiment_id = get_last_experiment()
        if command == "delete":
            experiment = get_entry("experiments", filter=dict(id=(1, experiment_id)))
            delete_entry(experiment)
        elif command == "export_ii":
            export_interface_interference_screenshots(experiment_id, verbose)
        elif command == "num_websites":
            show_num_websites(experiment_id, attr, verbose)
        elif command == "show_errors":
            show_errors(experiment_id)
        elif command == "clean":
            delete_websites_where_crawl_was_interrupted(experiment_id)
        else:
            raise ValueError(f"Unrecognized command {command}.")


if __name__ == "__main__":
    main()
