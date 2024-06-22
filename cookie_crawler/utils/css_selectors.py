from pathlib import Path
from typing import List, Tuple

import requests

from shared_utils import read_txt_file, write_to_file

GENERAL_SELECTORS_URL = "https://raw.githubusercontent.com/easylist/easylist/master/easylist_cookie/easylist_cookie_general_hide.txt"
SPECIFIC_SELECTORS_URL = "https://raw.githubusercontent.com/easylist/easylist/master/easylist_cookie/easylist_cookie_specific_hide.txt"
SELECTORS_EXCEPTIONS_URL = "https://raw.githubusercontent.com/easylist/easylist/master/easylist_cookie/easylist_cookie_allowlist_general_hide.txt"
BANNED_CHARACTERS = {"!", "[", "@", "|", "/"}


def parse_general_selectors(raw_selectors: List[str]) -> List[str]:
    selectors = []
    for line in raw_selectors:
        if line[0] in BANNED_CHARACTERS:
            continue
        if line[0:2] == "##":
            selector = line[2:]
            selectors.append(selector.strip())
    return selectors


def parse_custom_selectors(
    url: str, specific_selectors: List[str], category: str
) -> List[str]:
    assert category in [
        "specific",
        "exceptions",
    ], f"Unknown selectors category {category}"
    separator = "##" if category == "specific" else "#@#"
    selectors = []
    for line in specific_selectors:
        if line[0] in BANNED_CHARACTERS:
            continue
        if separator in line:
            domains, selector = line.split(separator)
            if any(s in url for s in domains.split(",")):
                selectors.append(selector)
    return selectors


def parse_selectors_for_url(
    url: str,
    general_selectors: List[str],
    specific_selectors: List[str],
    selectors_exceptions: List[str],
) -> List[str]:
    selectors = general_selectors.copy()
    selectors.extend(parse_custom_selectors(url, specific_selectors, "specific"))
    exceptions = parse_custom_selectors(url, selectors_exceptions, "exceptions")
    selectors = [selector for selector in selectors if selector not in exceptions]
    return selectors


def get_selectors() -> Tuple[List[str], ...]:
    selectors = []
    for desc, url in zip(
        ["general_selectors", "specific_selectors", "selectors_exceptions"],
        [GENERAL_SELECTORS_URL, SPECIFIC_SELECTORS_URL, SELECTORS_EXCEPTIONS_URL],
    ):
        filename = Path(f"config/selectors/{desc}.txt")
        if filename.is_file():
            print("Loading selectors from cache ..")
            tmp_selectors = read_txt_file(filename)
        else:
            print("Downloading selectors ..")
            filename.parent.mkdir(exist_ok=True, parents=True)
            tmp_selectors = requests.get(url).text.splitlines()
            write_to_file("\n".join(tmp_selectors), filename)
        selectors.append(tmp_selectors)
    selectors[0] = parse_general_selectors(selectors[0])
    return tuple(selectors)
