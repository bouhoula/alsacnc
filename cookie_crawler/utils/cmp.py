import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from selenium.webdriver import Firefox

from shared_utils import repeat


def get_action_targets(action: Dict) -> List[Dict]:
    targets = []
    for key, value in action.items():
        if key == "target":
            targets.append({"target": value})
            if "parent" in action:
                targets[-1]["parent"] = action["parent"]
        elif isinstance(value, dict):
            targets.extend(get_action_targets(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    targets.extend(get_action_targets(item))
    return targets


def get_consentomatic_rules(directories: List[str]) -> Tuple[Dict, List]:
    rules: Dict[str, Dict] = {}
    cmps: List[str] = []

    for directory in directories:
        for file_path in Path(directory).glob("*.json"):
            with file_path.open() as f:
                cmps.append(
                    Path(f.name).name.replace(".json", "").split("_")[0].lower()
                )
                rule = json.load(f)
                keys = [key for key in rule.keys() if key != "$schema"]
                assert len(keys) == 1
                cmp_name = keys[0]
                rules[cmp_name] = {}
                rules[cmp_name]["detectors"] = [
                    detector["presentMatcher"]
                    for detector in rule[cmp_name]["detectors"]
                ]
                rules[cmp_name]["actions_targets"] = []
                for action in rule[cmp_name]["methods"]:
                    rules[cmp_name]["actions_targets"].extend(
                        get_action_targets(action)
                    )

    return rules, cmps


@repeat(raise_exception_if_fails=False, return_value_if_fails=[])
def detect_cmp_consentomatic(webdriver: Firefox) -> List[str]:
    rules, cmps = get_consentomatic_rules(
        [f"{os.getenv('CONSENTOMATIC_DIR')}/rules", "config/custom_consentomatic_rules"]
    )
    return webdriver.execute_script(
        open("cookie_crawler/scripts/detect_cmp_consentomatic.js").read(),
        rules,
        cmps,
    )


@repeat(raise_exception_if_fails=False, return_value_if_fails={})
def detect_cmp_tcfapi(webdriver: Firefox) -> Dict:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/detect_cmp_tcfapi.js").read()
    )


def detect_cmp(
    website: Dict,
    webdriver: Firefox,
    detect_cmp_enabled: bool,
    detect_cmp_methods: List,
) -> None:
    if not detect_cmp_enabled:
        return
    if "cmp" not in website or website["cmp"] is None:
        website["cmp"] = {}
    for method in detect_cmp_methods:
        website["cmp"][method] = eval(f"detect_cmp_{method}")(webdriver)


if __name__ == "__main__":
    rules, cmps = get_consentomatic_rules(
        ["consentomatic/rules", "config/custom_consentomatic_rules"]
    )
    with open("ignored_by_git/rules.json", "w") as f:
        json.dump(rules, f)
    with open("ignored_by_git/cmps.json", "w") as f:
        json.dump(cmps, f)
