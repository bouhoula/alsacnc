import os
from typing import Dict, Optional, Tuple

import requests


def set_up_proxy(config: Dict, browser_params: Optional[Dict] = None) -> None:
    if not config["proxy_enabled"]:
        return

    if browser_params is not None:
        browser_params["prefs"]["network.proxy.type"] = 1
        browser_params["prefs"]["network.proxy.http"] = config["proxy_server"]
        browser_params["prefs"]["network.proxy.http_port"] = config["proxy_port"]
        browser_params["prefs"]["network.proxy.ssl"] = config["proxy_server"]
        browser_params["prefs"]["network.proxy.ssl_port"] = config["proxy_port"]
        browser_params["prefs"]["network.proxy.no_proxies_on"] = config["no_proxies_on"]

    proxy_url = f"{config['proxy_server']}:{config['proxy_port']}"
    os.environ["HTTP_PROXY"] = f"http://{proxy_url}"
    os.environ["HTTPS_PROXY"] = f"https://{proxy_url}"
    os.environ["NO_PROXY"] = config["no_proxies_on"]


def get_ip_location() -> Tuple[Optional[str], Optional[str]]:
    try:
        response = requests.get("http://lumtest.com/myip.json").json()
        country = response["country"]
        region = response["geo"]["region"] if response["geo"]["region"] != "" else None
        return country, region
    except:
        return None, None
