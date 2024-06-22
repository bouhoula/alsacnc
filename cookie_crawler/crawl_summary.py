from typing import Dict, List, Optional, Set, Tuple

import click
import pandas as pd

from database.queries import get_last_experiment, get_table, init_db

COOKIEBLOCK_THRESHOLD = 2


def get_ratio_string(value: int, base: int) -> str:
    return f"{value}/{base} ({value/base*100:.1f}%)" if base > 0 else f"{value}/0"


def get_violation(
    df: pd.DataFrame,
    violation_name: str,
    violation_base: int,
    violations: Dict,
    websites_with_violations: Optional[Set] = None,
) -> None:
    violations[violation_name] = get_ratio_string(len(df), violation_base)
    if websites_with_violations is not None:
        websites_with_violations.update(df["name"])


def get_violations(crawl_results: pd.DataFrame) -> Tuple[Dict, str, str]:
    violations: Dict = dict()
    websites_with_violations: Set = set()

    get_violation(
        crawl_results[
            (crawl_results["accept_button_detected_without_reject_button"] == True)
            & (crawl_results["tracking_detected"] >= COOKIEBLOCK_THRESHOLD)
        ],
        "missing_reject",
        len(crawl_results[crawl_results["accept_button_detected"] >= 1]),
        violations,
        websites_with_violations,
    )

    tracking_column_names = [
        f"tracking_detected_after_{consent_option}"
        for consent_option in ["reject", "close", "save"]
    ] + ["tracking_detected_prior_to_interaction"]

    base_column_names = [
        f"{consent_option}_button_detected"
        for consent_option in ["reject", "close", "save"]
    ] + ["cookie_notice_detected"]

    for column_name, base_column_name in zip(tracking_column_names, base_column_names):
        get_violation(
            crawl_results[crawl_results[column_name] >= COOKIEBLOCK_THRESHOLD],
            column_name.replace("tracking", "AA_cookies"),
            len(crawl_results[crawl_results[base_column_name] >= 1]),
            violations,
            websites_with_violations,
        )

    get_violation(
        crawl_results[
            (crawl_results["tracking_detected"] > COOKIEBLOCK_THRESHOLD)
            & (crawl_results["tracking_purposes_detected_in_initial_text"] == 0)
        ],
        "undeclared_purposes",
        len(crawl_results[crawl_results["cookie_notice_detected"] >= 1]),
        violations,
        websites_with_violations,
    )

    violations_with_cookie_notices = get_ratio_string(
        len(websites_with_violations),
        len(crawl_results[crawl_results["cookie_notice_detected"] >= 1]),
    )

    get_violation(
        crawl_results[
            (crawl_results["cookie_notice_detected"] == False)
            & (crawl_results["tracking_detected"] >= COOKIEBLOCK_THRESHOLD)
        ],
        "missing_notice",
        len(crawl_results[crawl_results["tracking_detected"] >= COOKIEBLOCK_THRESHOLD]),
        violations,
        websites_with_violations,
    )

    violations_total = get_ratio_string(
        len(websites_with_violations), len(crawl_results)
    )

    return violations, violations_with_cookie_notices, violations_total


def get_dark_patterns(crawl_results: pd.DataFrame) -> Dict:
    dark_patterns: Dict = dict()
    column_names = ["forced_action_detected", "interface_interference_detected"]
    base_values = [
        len(crawl_results[crawl_results["cookie_notice_detected"] >= 1]),
        len(
            crawl_results[
                (crawl_results["accept_button_detected"] == 1)
                & (
                    (crawl_results["reject_button_detected"] == 1)
                    | (crawl_results["close_button_detected"] == 1)
                    | (crawl_results["save_button_detected"] == 1)
                )
            ]
        ),
    ]
    for dp, base_value in zip(column_names, base_values):
        get_violation(
            crawl_results[crawl_results[dp] >= 1],
            dp.replace("_detected", ""),
            base_value,
            dark_patterns,
        )
    return dark_patterns


def print_dict(
    d: Dict, tab: int = 0, keys_of_interest: Optional[List[str]] = None
) -> None:
    for k, v in d.items():
        if keys_of_interest is None or k in keys_of_interest:
            print(" " * tab + f"{k}: {v}")


def filter_crawl_results_for_cmps(
    crawl_results: pd.DataFrame, cmps: List[str]
) -> pd.DataFrame:
    return crawl_results[
        crawl_results["consentomatic"].apply(
            lambda x: len(set(x).intersection(cmps)) > 0
        )
    ]


@click.command()
@click.option("--experiment_id", default=None)
def main(experiment_id: Optional[str]) -> None:
    init_db("postgres", create_tables=True)

    # Important: update experiment_id if it does not correspond to the last experiment
    # Use `python database/queries.py --command ls` to display all experiment
    if experiment_id is None:
        experiment_id = get_last_experiment()

    domains_path = "domains/crux_202303_eu_uk_top_10000_N_10000.csv"

    print("Number of websites in the crawling list:", len(pd.read_csv(domains_path)))

    supported_languages = [
        "da",
        "de",
        "en",
        "es",
        "fi",
        "fr",
        "it",
        "nl",
        "pl",
        "pt",
        "sv",
    ]

    filter_dict = dict(experiment_id=(1, experiment_id))
    websites = get_table("websites", filter=filter_dict)

    total_num_websites = len(websites)
    print("Number of websites with logged entries: ", total_num_websites)

    num_websites_with_unsupported_language = len(
        websites[
            (~websites["language"].isna())
            & (~websites["language"].isin(supported_languages))
        ]
    )
    print(
        "Number of websites with an unsupported language:",
        get_ratio_string(num_websites_with_unsupported_language, total_num_websites),
    )

    num_websites_with_supported_language = len(
        websites[websites["language"].isin(supported_languages)]
    )
    print(
        "Number of websites with an supported language:",
        get_ratio_string(num_websites_with_supported_language, total_num_websites),
    )

    errors = get_table(
        "websites",
        mode="join",
        subquery_table="errors",
        subquery_col="website_id",
        filter=filter_dict,
    )

    crawl_results = get_table(
        "websites",
        mode="join",
        subquery_table="crawl_results",
        subquery_col="website_id",
        filter=filter_dict,
    )

    crawl_results = crawl_results[crawl_results["language"].isin(supported_languages)]
    crawl_results = crawl_results[
        ~crawl_results["website_id"].isin(errors["website_id"])
    ]
    num_websites_successful = len(crawl_results)
    print(
        "Number of websites crawled successfully:",
        get_ratio_string(num_websites_successful, num_websites_with_supported_language),
    )

    (
        violations,
        num_violations_with_cookies_notices,
        num_violations_total,
    ) = get_violations(crawl_results)

    print(
        "Number of websites with cookie notices with at least one violation:",
        num_violations_with_cookies_notices,
    )
    print("Number of websites with at least one violation:", num_violations_total)
    print("Violation details:")
    print_dict(violations, tab=2)
    dark_patterns = get_dark_patterns(crawl_results)
    print("Dark pattern details")
    print_dict(dark_patterns, tab=2)

    print("\nRank-based results")
    for rank in sorted(crawl_results["crux_rank"].unique()):
        print("  Rank:", rank)
        violations, _, _ = get_violations(
            crawl_results[crawl_results["crux_rank"] == rank]
        )
        print_dict(violations, tab=4)
        dark_patterns = get_dark_patterns(
            crawl_results[crawl_results["crux_rank"] == rank]
        )
        print_dict(dark_patterns, tab=4)

    crawl_results["consentomatic"] = crawl_results["cmp"].apply(
        lambda x: [] if x is None else x["consentomatic"]
    )
    crawl_results["uses_tcf_iab"] = crawl_results["cmp"].apply(
        lambda x: x is not None
        and "tcfapi" in x
        and x["tcfapi"] is not None
        and "cmpId" in x["tcfapi"]
    )

    print("\nCMP-based results")
    print("  Websites using TCF IAB:")
    print_dict(
        get_violations(crawl_results[crawl_results["uses_tcf_iab"]])[0],
        tab=4,
        keys_of_interest=[
            "AA_cookies_detected_after_reject",
            "AA_cookies_detected_prior_to_interaction",
        ],
    )
    print(
        "  Websites analyzed by Bollinger et al. Cookiebot + Onetrust + Termly (Only Cookiebot for 'Ignored Reject')"
    )
    print_dict(
        get_violations(
            filter_crawl_results_for_cmps(
                crawl_results, ["onetrust", "cookiebot", "termly"]
            )
        )[0],
        tab=4,
        keys_of_interest=["AA_cookies_detected_prior_to_interaction"],
    )
    # Ignored reject violation only applied to Cookiebot websites in Bollinger et al.
    print_dict(
        get_violations(filter_crawl_results_for_cmps(crawl_results, ["cookiebot"]))[0],
        tab=4,
        keys_of_interest=["AA_cookies_detected_after_reject"],
    )

    print(
        "  Websites analyzed by Nouwens et al. Cookiebot + Onetrust + Crownpeak + Quantcast + Trustarc"
    )
    nouwens_subset = filter_crawl_results_for_cmps(
        crawl_results, ["onetrust", "cookiebot", "crownpeak", "trustarc", "quantcast"]
    )
    print_dict(
        get_violations(nouwens_subset)[0],
        tab=4,
        keys_of_interest=["AA_cookies_detected_prior_to_interaction"],
    )
    print_dict(
        get_dark_patterns(nouwens_subset), tab=4, keys_of_interest=["forced_action"]
    )


if __name__ == "__main__":
    main()
