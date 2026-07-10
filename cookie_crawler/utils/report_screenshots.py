import json
import logging
import os
import re
from typing import List, Optional

from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement

from cookie_crawler.utils.js import (
    highlight_element,
    remove_highlights,
    scroll_into_view,
)

logger = logging.getLogger("openwpm")

MANIFEST_NAME = "report_screenshots.json"


def strategy_to_filename_part(strategy: str) -> str:
    part = re.sub(r"[^a-z0-9]+", "_", strategy.lower()).strip("_")
    return part or "strategy"


def _manifest_path(save_path: str) -> str:
    return os.path.join(save_path, MANIFEST_NAME)


def _append_to_manifest(save_path: str, entry: dict) -> None:
    path = _manifest_path(save_path)
    manifest: List[dict] = []
    if os.path.isfile(path):
        try:
            with open(path) as fin:
                manifest = json.load(fin)
        except (json.JSONDecodeError, OSError):
            manifest = []
    manifest.append(entry)
    with open(path, "w") as fout:
        json.dump(manifest, fout, indent=2)


def save_report_screenshot(
    webdriver: Firefox,
    save_path: str,
    strategy: str,
    step: int,
    highlighted_text: Optional[str] = None,
) -> None:
    filename = f"report_{strategy_to_filename_part(strategy)}_step{step:02d}.png"
    try:
        png = webdriver.get_screenshot_as_png()
        with open(os.path.join(save_path, filename), "wb") as fout:
            fout.write(png)
    except Exception as e:  # noqa: BLE001 - never let a screenshot break a crawl
        logger.warning(f"Could not save report screenshot {filename}: {e}")
        return
    _append_to_manifest(
        save_path,
        dict(
            strategy=strategy,
            step=step,
            highlighted_text=highlighted_text,
            filename=filename,
        ),
    )


def highlight_and_capture(
    webdriver: Firefox,
    element: WebElement,
    save_path: str,
    strategy: str,
    step: int,
    highlighted_text: Optional[str] = None,
    color: str = "#ff0000",
) -> None:
    try:
        scroll_into_view(element, webdriver)
    except Exception:  # noqa: BLE001
        pass
    highlight_element(element, webdriver, color)
    save_report_screenshot(webdriver, save_path, strategy, step, highlighted_text)
    remove_highlights(webdriver)
