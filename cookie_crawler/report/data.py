import base64
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from cookie_crawler.utils.report_screenshots import MANIFEST_NAME
from database.queries import get_table

STAGE_STRATEGIES = {
    "avant_toute_action": "No interaction",
    "apres_refus": "Reject",
    "apres_acceptation": "Accept",
    "apres_retrait": "Close/Continue without accepting",
}

def _get_website_row(experiment_id: str, website_name: str) -> Optional[Dict[str, Any]]:
    df = get_table(
        "websites",
        filter={"experiment_id": (1, experiment_id), "name": (1, website_name)},
    )
    if df is None or len(df) == 0:
        return None
    return df.iloc[0].to_dict()


def list_websites(experiment_id: str) -> List[str]:
    df = get_table("websites", filter={"experiment_id": (1, experiment_id)})
    if df is None or len(df) == 0:
        return []
    return sorted(df["name"].dropna().unique().tolist())


def _is_first_party(cookie_domain: Optional[str], website_name: str) -> bool:
    cookie_domain = (cookie_domain or "").lstrip(".")
    if not (cookie_domain and website_name):
        return False
    return website_name in cookie_domain or cookie_domain in website_name


def _cookies_by_strategy(
    website_id: int,
    website_name: str,
    exclude_first_party: bool,
) -> Dict[str, List[Dict[str, str]]]:
    """Return {strategy: [{nom, valeur}, ...]} from the cookies tables."""
    df: Optional[pd.DataFrame] = None
    for table_name in ("cookies_with_predictions", "javascript_cookies"):
        try:
            candidate = get_table(table_name, filter={"website_id": (1, website_id)})
        except Exception:
            candidate = None
        if candidate is not None and len(candidate) > 0:
            df = candidate
            break
    result: Dict[str, List[Dict[str, str]]] = {}
    if df is None or len(df) == 0:
        return result
    seen: Dict[str, set] = {}
    for _, row in df.iterrows():
        strategy = row.get("collection_strategy")
        name = row.get("name")
        value = row.get("value")
        domain = row.get("cookie_domain")
        if not strategy or not name:
            continue
        if exclude_first_party and _is_first_party(domain, website_name):
            continue
        key = (name, value)
        if key in seen.setdefault(strategy, set()):
            continue
        seen[strategy].add(key)
        result.setdefault(strategy, []).append({"nom": name, "valeur": value or ""})
    return result


def _caption(highlighted_text: Optional[str]) -> str:
    if highlighted_text:
        return f"Clic sur « {highlighted_text} »"
    return ""


def _load_screenshots(save_path: str, strategy: str) -> List[Dict[str, str]]:
    """Read the manifest, return ordered [{src, caption}] for a strategy."""
    if not save_path:
        return []
    manifest_path = os.path.join(save_path, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return []
    try:
        with open(manifest_path) as fin:
            manifest = json.load(fin)
    except (json.JSONDecodeError, OSError):
        return []
    entries = [e for e in manifest if e.get("strategy") == strategy]
    entries.sort(key=lambda e: e.get("step", 0))
    captures: List[Dict[str, str]] = []
    for entry in entries:
        filename = entry.get("filename")
        if not filename:
            continue
        png_path = os.path.join(save_path, filename)
        if not os.path.isfile(png_path):
            continue
        with open(png_path, "rb") as fin:
            b64 = base64.b64encode(fin.read()).decode("utf-8")
        captures.append(
            {
                "src": f"data:image/png;base64,{b64}",
                "caption": _caption(entry.get("highlighted_text")),
            }
        )
    return captures


def build_constats(
    experiment_id: str,
    website_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    exclude_first_party: bool = True,
) -> Dict[str, Any]:
    """Build the full context dict for `template.jinja2` for one website."""
    metadata = metadata or {}
    website = _get_website_row(experiment_id, website_name)
    if website is None:
        raise ValueError(
            f"No website '{website_name}' found for experiment '{experiment_id}'."
        )
    website_id = website["id"]
    save_path = website.get("save_path") or ""

    crawl_df = get_table("crawl_results", filter={"website_id": (1, website_id)})
    if crawl_df is not None and len(crawl_df) > 0:
        crawl = crawl_df.iloc[0].to_dict()
    else:
        crawl = {}
    cookie_notice_detected = bool(crawl.get("cookie_notice_detected"))
    reject_detected = bool(crawl.get("reject_button_detected"))
    close_detected = bool(crawl.get("close_button_detected"))

    cookies = _cookies_by_strategy(website_id, website_name, exclude_first_party)

    constats: Dict[str, Any] = {
        "navigateur": metadata.get("navigateur", {}),
        "fin": metadata.get("fin", {}),
        "absence_bandeau": not cookie_notice_detected,
        "absence_refus": not reject_detected,
        "absence_close": not close_detected,
    }
    for stage, strategy in STAGE_STRATEGIES.items():
        constats[stage] = {
            "captures": _load_screenshots(save_path, strategy),
            "cookies": cookies.get(strategy, []),
        }

    data: Dict[str, Any] = {"pv_num": metadata.get("pv_num", "")}
    data["header"] = metadata.get("header", {})
    data["footer"] = metadata.get("footer", {})
    data["agents"] = metadata.get("footer", {}).get("agents", [])
    data["url"] = website_name
    data["constats"] = constats
    return data
