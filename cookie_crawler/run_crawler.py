import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager

import click

from cookie_crawler.commands.cookie_banner_command import GetCookieBannerCommand
from cookie_crawler.utils.callbacks import get_callback
from cookie_crawler.utils.css_selectors import get_selectors
from cookie_crawler.utils.domains import get_domains
from cookie_crawler.utils.monkey_patches import (
    apply_monkey_patch_to_task_manager,
    apply_monkey_patches,
    set_screen_dimensions,
)
from cookie_crawler.utils.monkey_patches.command_sequence import CommandSequence
from cookie_crawler.utils.proxy import get_ip_location, set_up_proxy
from database.queries import (
    check_for_timeout,
    delete_entry,
    get_entry,
    init_db,
    insert_into_db,
    update_entry,
)
from shared_utils import (
    click_options_from_config,
    load_yaml,
    override_config,
    url_to_uniform_domain,
)

CONFIG_TEMPLATE_PATH = "config/experiment_config.yaml"

logger = logging.getLogger("openwpm")


@click.command()
@click.option("--config_path", default=None)
@click_options_from_config(
    CONFIG_TEMPLATE_PATH, keys_to_ignore=["browser_config", "origins"]
)
# flake8: noqa: C901
def main(config_path: str, **kwargs: Any) -> None:
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M")

    if config_path is None:
        config_path = CONFIG_TEMPLATE_PATH
    config = load_yaml(config_path)
    override_config(config, **kwargs)

    if config["debug"]:
        os.environ["DB_NAME"] = "cookie_db_debug"
    init_db(
        engine_name=config["engine"],
        create_tables=True,
        drop_existing_tables=config["clear_db"],
    )
    print("initialized database")

    experiment_id = config["experiment_id"]
    if experiment_id is None:
        if config["debug"]:
            experiment_id = "debug"
        else:
            domains_path = config["domains_config"]["domains_path"]
            if domains_path is not None:
                experiment_id = Path(domains_path).with_suffix("").name
            else:
                countries_desc = config["domains_config"]["countries_desc"]
                origin = config["origin"]
                if countries_desc is None:
                    countries_desc = "_".join(config["domains_config"]["country"])
                experiment_id = f"crux_{countries_desc}_{origin}"
            experiment_id = f"{experiment_id}_{start_time_str}"

    experiment = get_entry("experiments", filter=dict(id=(1, experiment_id)))
    if experiment is None:
        experiment = insert_into_db(
            "experiments", dict(id=experiment_id, config=config)
        )

    config = config.copy()

    apply_monkey_patches()
    set_screen_dimensions(config["screen_width"], config["screen_height"])

    save_intervals: Any
    websites, save_intervals = get_domains(**config["domains_config"])
    if config["debug"]:
        websites = websites[:1]
        config["num_browsers"] = 1

    if len(websites) == 0:
        print("Website crawl list is empty. Aborting...", file=sys.stderr)
        return
    else:
        print(f"Crawling {len(websites)} websites ...")

    manager_params = ManagerParams(
        num_browsers=config["num_browsers"], _failure_limit=32000
    )
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    manager_params.log_path = log_dir / f"crawl_{start_time_str}.log"
    manager_params.log_path.touch(exist_ok=True)
    manager_params.data_directory = Path(f"./experiments/{experiment_id}")
    manager_params.data_directory.mkdir(exist_ok=True, parents=True)

    if config["engine"] == "sqlite":
        os.environ["DB_PATH"] = str(
            manager_params.data_directory / "cookie_banners.sqlite"
        )

    selectors = get_selectors()
    browser_params = list()
    browser_params_dict = config.pop("browser_config")
    set_up_proxy(config["proxy"], browser_params_dict)

    country, region = get_ip_location()
    if country is None:
        print("Warning: could not determine IP location.")
    elif experiment["country"] is not None and country != experiment["country"]:
        print(
            f"Warning: experiment {experiment['id']} was initially started with a "
            f"{experiment['country']} IP and is being resumed with a {country} IP"
        )
    else:
        print(f"IP location: {f'{region}, ' if region is not None else ''}{country}.")
    time.sleep(5)

    experiment["region"] = region
    experiment["country"] = country
    update_entry(experiment)

    for _ in range(config["num_browsers"]):
        browser_params.append(BrowserParams.from_dict(browser_params_dict))
        browser_params[-1].custom_params.update(config)
        browser_params[-1].custom_params["selectors"] = selectors

    openwpm_db_path = manager_params.data_directory / "crawl.sqlite"
    storage_provider = SQLiteStorageProvider(openwpm_db_path)
    os.environ["OPENWPM_DB_PATH"] = str(openwpm_db_path)
    max_num_attempts = (
        experiment["num_full_iterations"] + 1
        if config["override"]
        else config["num_attempts_per_website"]
    )

    if "other" not in experiment:
        experiment["other"] = dict()

    if config["save_intervals"] is not None:
        save_intervals = config["save_intervals"]
    if isinstance(save_intervals, int):
        save_intervals = [
            (i, i + save_intervals) for i in range(0, len(websites), save_intervals)
        ]
    elif isinstance(save_intervals, list) and isinstance(save_intervals[0], int):
        save_intervals = [
            (0 if i == 0 else save_intervals[i - 1], save_intervals[i])
            for i in range(len(save_intervals))
        ]
    if save_intervals is None or save_intervals[0][1] > len(websites):
        save_intervals = [(0, len(websites))]
    save_intervals = save_intervals[config["first_interval"] :]
    if max_num_attempts == experiment["num_full_iterations"]:
        max_num_attempts += 1

    for num_iteration in range(experiment["num_full_iterations"], max_num_attempts):
        for segment_start, segment_end in save_intervals:
            print(f"Crawling websites between {segment_start} and {segment_end}")
            websites_segment = websites[segment_start:segment_end]
            manager = TaskManager(
                manager_params, browser_params, storage_provider, None
            )
            apply_monkey_patch_to_task_manager(manager)
            websites_names = []

            for url, crux_top_country, crux_top_rank, crux_ranks in websites_segment:
                website_name = url_to_uniform_domain(url)
                website = get_entry(
                    "websites",
                    filter=dict(
                        experiment_id=(1, experiment_id), name=(1, website_name)
                    ),
                )
                previously_timed_out = 0
                websites_names.append(website_name)
                if website is not None:
                    timeout_check = check_for_timeout(website)
                    if not config["override"] and (
                        website["success"] == 1 or timeout_check != "No timeout"
                    ):
                        continue
                    if timeout_check == "Timeout during exploration":
                        previously_timed_out = 1
                    delete_entry(website)
                    print("deleted website")

                website_save_path = (
                    manager_params.data_directory / "data" / website_name
                )
                website_save_path.mkdir(exist_ok=True, parents=True)

                website = insert_into_db(
                    "websites",
                    dict(
                        name=website_name,
                        url=url,
                        save_path=str(website_save_path),
                        crux_rank=crux_top_rank,
                        crux_country=crux_top_country,
                        ranking_data=dict(crux=crux_ranks),
                        previously_timed_out=previously_timed_out,
                        experiment_id=experiment["id"],
                    ),
                )
                if not isinstance(website, dict):
                    print(
                        "Warning: `insert_into_database` did not return a dictionary "
                        f"as expected. Skipping website {website_name}"
                    )
                    continue

                callback = get_callback(website)
                command_sequence = CommandSequence(
                    website,
                    reset=True,
                    blocking=False,
                    callback=callback,
                )
                command_sequence.append_command(
                    GetCookieBannerCommand(website, config),
                    timeout=config["timeout"],
                )
                manager.execute_command_sequence(command_sequence)
            manager.close()

        experiment["num_full_iterations"] += 1
        update_entry(experiment)

    print(
        "Total running time:", timedelta(seconds=time.time() - start_time.timestamp())
    )


if __name__ == "__main__":
    main()
