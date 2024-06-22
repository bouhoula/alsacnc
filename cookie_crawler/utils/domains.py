import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import click
import pandas as pd
from google.cloud import bigquery
from tranco import Tranco

from shared_utils import (
    click_options_from_config,
    load_yaml,
    read_txt_file,
    write_to_file,
)


def get_tranco_domains(
    date: Optional[str] = None,
    list_id: Optional[str] = None,
    num_domains: Optional[int] = None,
    extension: Optional[str] = None,
    sample: Optional[Tuple[int, int, int]] = None,
    output_file: Optional[str] = None,
) -> List[str]:
    assert date is not None or list_id is not None
    domains = (
        Tranco(cache=True, cache_dir="domains/").list(date=date, list_id=list_id).top()
    )
    if extension is not None:
        if extension[0] != ".":
            extension = f".{extension}"
        domains = [domain for domain in domains if domain.endswith(extension)]
    if num_domains is not None and num_domains < len(domains):
        domains = domains[:num_domains]
    if sample is not None:
        num_samples, st, en = sample
        domains = domains[st:en]
        domains = random.sample(domains, num_samples)
    if output_file is not None:
        write_to_file("\n".join(domains), output_file)
    return domains


def set_to_df(
    sample: Set,
    df: pd.DataFrame,
    countries: Tuple[str, ...],
    sampled_from: Dict[str, List[str]],
) -> pd.DataFrame:
    res = []
    df_sample = df[df["origin"].isin(sample)]
    for origin, df_origin in df_sample.groupby(df_sample["origin"]):
        ranks = {country: None for country in countries}
        lowest_rank = float("inf")
        lowest_country = "zz"
        for row in df_origin.to_dict("records"):
            ranks[row["country_code"]] = row["rank"]
            if row["rank"] < lowest_rank or (
                row["rank"] == lowest_rank
                and row["country_code"] < lowest_country
                and row["country_code"] in countries
            ):
                lowest_rank = row["rank"]
                lowest_country = row["country_code"]
        res.append(
            {
                "origin": origin,
                "top_rank_country": lowest_country,
                "top_rank": lowest_rank,
                "ranks": ranks,
                "sampled_from": sampled_from[origin],
            }
        )
    return pd.DataFrame(res)


# flake8: noqa: C901
def get_crux_domains(
    countries: Union[str, Tuple[str, ...]],
    countries_desc: Optional[str],
    maximum_rank: Union[int, Tuple[int, ...]],
    date: str,
    domains_path: Optional[str] = None,
    num_samples_per_country_per_rank: Optional[Union[int, Dict[str, Any]]] = None,
    split_interval: int = 0,
    sample_from: Optional[List[str]] = None,
    redownload_data: bool = False,
) -> Tuple[List[Tuple], List[int]]:
    if isinstance(countries, str):
        countries = (countries,)
    else:
        countries = tuple(countries)

    if not isinstance(maximum_rank, int):
        maximum_rank = tuple(maximum_rank)

    if domains_path is None:
        if countries_desc is None:
            countries_desc = "_".join(countries)
        domains_path = f"domains/crux_{date}_{countries_desc}"
        if isinstance(maximum_rank, int):
            domains_path += f"_top_{maximum_rank}"
        else:
            domains_path += f"_ranks_{'_'.join(map(str, maximum_rank))}"
        if isinstance(num_samples_per_country_per_rank, int):
            domains_path += f"_N_{num_samples_per_country_per_rank}"
        domains_path += ".csv"
    save_intervals_path = domains_path.replace(".csv", "_save_intervals.json")

    if redownload_data or not Path(domains_path).is_file():
        print("Downloading CrUX data to", domains_path)

        client = bigquery.Client()
        country_condition = (
            f"country_code in {countries}"
            if len(countries) > 1
            else f"country_code = '{countries[0]}'"
        )
        rank_condition = (
            f"experimental.popularity.rank <= {maximum_rank}"
            if isinstance(maximum_rank, int)
            else f"experimental.popularity.rank in {maximum_rank}"
        )

        query = f"""SELECT DISTINCT origin, country_code, experimental.popularity.rank as rank
                FROM `chrome-ux-report.experimental.country`
                WHERE yyyymm = {date}
                AND {country_condition}
                AND {rank_condition}"""

        df = client.query(query).to_dataframe()
        df = df.sort_values(by=["origin", "rank", "country_code"])

        random.seed(0)
        sample_dfs = []
        save_intervals = []
        num_samples_per_iter: Any
        sampled_from: Dict[str, List[str]] = {}
        if num_samples_per_country_per_rank is not None:
            if isinstance(num_samples_per_country_per_rank, int):
                if split_interval > 0:
                    assert num_samples_per_country_per_rank % split_interval == 0
                    num_iter = num_samples_per_country_per_rank // split_interval
                    num_samples_per_iter = split_interval
                else:
                    num_iter = 1
                    num_samples_per_iter = num_samples_per_country_per_rank
            else:
                num_samples_per_iter = dict()
                num_iter = 1
                for rank_i, rank in enumerate(sorted(df["rank"].unique())):
                    num_samples_per_iter[rank] = dict()
                    for country in num_samples_per_country_per_rank:
                        num_samples_per_iter[rank][country] = (
                            num_samples_per_country_per_rank[country][rank_i]
                            if rank_i < len(num_samples_per_country_per_rank[country])
                            else 0
                        )

            per_country_out_global: Dict[str, List] = {
                country: [] for country in countries
            }
            out_global: List[str] = list()
            for _ in range(num_iter):
                out: Set = set()
                per_country_out: Dict[str, List] = {
                    country: [] for country in countries
                }
                for rank in sorted(df["rank"].unique()):
                    for country in countries:
                        tmp_num_samples = (
                            num_samples_per_iter
                            if isinstance(num_samples_per_iter, int)
                            else num_samples_per_iter[rank][country]
                        )

                        if tmp_num_samples > 0:
                            source = set(
                                df[
                                    (df["country_code"] == country)
                                    & (df["rank"] == rank)
                                    & (
                                        ~df["origin"].isin(
                                            per_country_out_global[country]
                                        )
                                    )
                                ]["origin"].values
                            )
                            sample = set(
                                random.sample(
                                    tuple(source),
                                    min(tmp_num_samples, len(source)),
                                )
                            )
                        else:
                            sample = set()
                        print(
                            f"New samples for rank={rank} country={country}: {len(sample)}"
                        )
                        per_country_out[country].extend(sample)
                        for domain in sample:
                            if domain not in sampled_from:
                                sampled_from[domain] = []
                            sampled_from[domain].append(country)
                        out = out | sample
                out = out - set(out_global)
                out_global.extend(out)
                sample_df = set_to_df(out, df, countries, sampled_from)
                sample_df = sample_df.sample(frac=1, random_state=0)
                save_intervals.append(len(sample_df))
                sample_dfs.append(sample_df)
                for country in countries:
                    per_country_out_global[country].extend(per_country_out[country])
            for i in range(1, len(save_intervals)):
                save_intervals[i] += save_intervals[i - 1]
        else:
            df["ranks"] = None

        df = pd.concat(sample_dfs)
        df.to_csv(domains_path, index=False)
        with open(save_intervals_path, "w") as fout:
            json.dump(save_intervals, fout)
    else:
        print("Using cached CrUX data", domains_path)
        df = pd.read_csv(domains_path)
        with open(save_intervals_path, "r") as fin:
            save_intervals = json.load(fin)
        df["ranks"] = df["ranks"].apply(lambda x: ast.literal_eval(x))
        df["sampled_from"] = df["sampled_from"].apply(lambda x: ast.literal_eval(x))

    if sample_from is not None:
        df = df[df["sampled_from"].apply(lambda x: any(k in sample_from for k in x))]

    return (
        list(
            map(
                lambda x: (x[0], x[1], int(x[2]), x[3]),
                df.itertuples(index=False, name=None),
            )
        ),
        save_intervals,
    )


def get_domains(
    domains_source: str,
    domains_path: Optional[str],
    countries: Tuple[str, ...],
    countries_desc: Optional[str],
    maximum_rank: Optional[int],
    crux_date: Optional[str],
    num_domains: int,
    num_samples_per_country_per_rank: Optional[int],
    split_interval: int,
    sample_from: Optional[List[str]] = None,
) -> Tuple[List[Tuple], Optional[List[int]]]:
    save_intervals = None
    if domains_source == "crux":
        assert maximum_rank is not None and crux_date is not None
        websites, save_intervals = get_crux_domains(
            countries,
            countries_desc,
            maximum_rank,
            crux_date,
            domains_path,
            num_samples_per_country_per_rank,
            split_interval,
            sample_from,
        )
    else:
        assert domains_path is not None
        websites = [(website, "", 0, None) for website in read_txt_file(domains_path)]

    if 0 < num_domains <= len(websites):
        websites = random.sample(websites, num_domains)

    return websites, save_intervals


def get_prefix_list(site: str) -> List[str]:
    if site.startswith("http"):
        return [""]
    elif site.startswith("www"):
        return ["https://", "http://"]
    else:
        return ["http://", "https://", "http://www.", "https://www."]


def is_url_to_different_domain(cur_url: Optional[str], origin_url: str) -> bool:
    if cur_url is None or not (cur_url.startswith("http") or cur_url.startswith("//")):
        return False
    cur_domain = urlparse(cur_url).hostname
    origin_domain = urlparse(origin_url).hostname
    return cur_domain != origin_domain


CONFIG_TEMPLATE_PATH = "config/experiment_config.yaml"


@click.command()
@click.option("--config_path", default=None)
@click_options_from_config(CONFIG_TEMPLATE_PATH, keys_to_ignore=["browser_config"])
def main(config_path: str, **kwargs: Any) -> None:
    config = load_yaml(CONFIG_TEMPLATE_PATH)
    websites, save_intervals = get_domains(**config["domains_config"])
    print(len(websites), save_intervals)


if __name__ == "__main__":
    main()
