import functools
import json
import logging
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import click
import yaml


def read_txt_file(filename: Union[str, Path]) -> List[str]:
    if not isinstance(filename, str):
        filename = str(filename)
    with open(filename) as fin:
        lines = fin.readlines()
    return [line.strip() for line in lines]


def load_yaml(filename: str) -> Dict[str, Any]:
    with open(filename, "r") as fin:
        args = yaml.safe_load(fin)
    return args


def write_to_file(
    s: Union[str, bytes], save_path: Union[Path, str], save_path_suffix: str = ""
) -> None:
    if isinstance(s, str):
        s = s.encode("utf8")
    with open(f"{save_path}{save_path_suffix}", "wb") as f:
        f.write(s)
        f.write(b"\n")


def dump_json(d: Dict, save_path: Union[Path, str]) -> None:
    with open(save_path, "w") as f:
        json.dump(d, f)


def strip(text: str) -> str:
    return re.sub("\\s{2,}", " ", text.strip())


def remove_duplicates(seq: List) -> List:
    new_seq = []
    for e in seq:
        if e not in new_seq:
            new_seq.append(e)
    return new_seq


def get_timestamp(s: Optional[str] = None) -> str:
    pattern = "[%Y-%m-%d %H:%M]" if s is None else f"[%Y-%m-%d %H:%M, {s}]"
    return datetime.now().strftime(pattern)


def repeat(
    num_tries: int = 3,
    sleep: int = 3,
    raise_exception_if_fails: bool = True,
    return_value_if_fails: Any = None,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            exception: Optional[Exception] = None
            for _ in range(num_tries):
                try:
                    return function(*args, **kwargs)
                except Exception as e:
                    time.sleep(sleep)
                    exception = e
            if raise_exception_if_fails:
                assert exception is not None
                msg = f"Execution of function {function.__name__} raised the following exception:\n {exception}. {traceback.format_exc()}"
                try:
                    raise exception.__class__(msg)
                except:
                    raise Exception(msg)
            else:
                return return_value_if_fails

        return wrapper

    return decorator


def set_global_logging_level(modules: List[str], level: int = logging.ERROR) -> None:
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefixes: list of one or more str prefixes to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    try:
        map(__import__, modules)
    except:
        pass
    prefix_re = re.compile(rf'^(?:{ "|".join(modules) })')
    for name in logging.root.manager.loggerDict:  # type: ignore
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def click_options_from_config(
    config_template: Union[str, Dict], keys_to_ignore: Optional[List[str]] = None
) -> Callable:
    if isinstance(config_template, str):
        config_template = load_yaml(config_template)
    if keys_to_ignore is None:
        keys_to_ignore = []

    def decorator(f: Callable) -> Callable:
        assert isinstance(config_template, dict)
        assert isinstance(keys_to_ignore, list)
        for key, value in config_template.items():
            if "." in key or key in keys_to_ignore:
                continue
            if isinstance(value, dict):
                f = click_options_from_config(value)(f)
            if isinstance(value, bool):
                f = click.option(f"--{key}/--no_{key}", default=None)(f)
            elif isinstance(value, (list, tuple)):
                f = click.option(f"--{key}", multiple=True)(f)
            elif isinstance(value, (int, float)):
                f = click.option(f"--{key}", type=type(value), default=None)(f)
            elif value is None or isinstance(value, str):
                f = click.option(f"--{key}", default=None)(f)
        return f

    return decorator


def override_config(config: Dict[str, Any], **kwargs: Any) -> None:
    for key, value in config.items():
        if isinstance(value, dict):
            override_config(value, **kwargs)
        elif key not in kwargs:
            continue
        else:
            new_value = kwargs[key]
            if new_value is None or (
                isinstance(new_value, tuple) and len(new_value) == 0
            ):
                continue
            config[key] = new_value


def url_to_uniform_domain(url: str) -> str:
    """Takes a URL or a domain string and transforms it into a uniform format.
    Examples: {"www.example.com", "https://example.com/", ".example.com"} --> "example.com"
    :param url: URL to clean and bring into uniform format
    """
    new_url = url.strip()
    new_url = re.sub("^http(s)?://", "", new_url)
    new_url = re.sub("^www([0-9])?", "", new_url)
    new_url = re.sub("^\\.", "", new_url)
    new_url = re.sub("/.*", "", new_url)
    return new_url


def capitalize(s: str) -> str:
    return " ".join(s.split("_")).capitalize()
