import json
from typing import Dict

import urllib3

from cookie_crawler.utils.domains import get_prefix_list


def detect_gpc(url: str) -> bool:
    prefix_list = get_prefix_list(url)
    for prefix in prefix_list:
        try:
            gpc_json_url = f"{prefix}{url.rstrip('/')}/.well-known/gpc.json"
            response = urllib3.PoolManager(timeout=5).request("GET", gpc_json_url)
            if response.status != 200:
                continue
            data = json.loads(response.data.decode())
            if "gpc" in data:
                return data["gpc"]
        except:
            continue
    return False


def enable_gpc(browser_params: Dict) -> None:
    browser_params["prefs"]["privacy.globalprivacycontrol.enabled"] = True
    browser_params["prefs"]["privacy.globalprivacycontrol.functionality.enabled"] = True