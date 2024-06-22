import binascii
import random
import time
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openwpm.config import BrowserParams

import numpy as np
import scipy
import scipy.cluster
from PIL import Image
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
)
from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement

from cookie_crawler.commands.get_command import reload_page_and_click_on_elements
from cookie_crawler.utils.colors import ColorDB
from cookie_crawler.utils.js import (
    element_is_hidden,
    find_element_by_selector,
    get_selector_from_element,
    get_visible_elements,
    scroll_into_view,
)

color_db = ColorDB()


def get_dominant_color(
    element: WebElement, num_clusters: int = 3
) -> Optional[Tuple[int, ...]]:
    img_path = f"tmp/element_{np.random.randint(1e15)}.png"
    Path(img_path).parent.mkdir(exist_ok=True, parents=True)
    if not element.screenshot(img_path):
        return None
    img = Image.open(img_path)
    img = img.resize((150, 150))
    arr = np.array(img)
    arr = arr.reshape(np.product(arr.shape[:2]), arr.shape[2]).astype(float)
    codes, _ = scipy.cluster.vq.kmeans(arr, num_clusters)
    vecs, _ = scipy.cluster.vq.vq(arr, codes)
    counts, _ = np.histogram(vecs, len(codes))
    color = codes[np.argmax(counts)]
    return tuple(map(int, color[:3]))


def convert_color_to_hex(color: Tuple[int, ...]) -> str:
    return f"#{binascii.hexlify(bytearray(int(c) for c in color)).decode('ascii')[:6]}"


def matching_colors(v1: Tuple[int, ...], v2: Tuple[int, ...]) -> bool:
    assert len(v1) == len(v2)
    return bool(np.max(np.abs((np.array(v1) - np.array(v2)))) < 255 // 4)


def get_font_properties(
    element: WebElement, webdriver: Firefox
) -> Tuple[int, int, str]:
    font_size, font_weight, text_color = webdriver.execute_script(
        """style=document.defaultView.getComputedStyle(arguments[0]);
        return [style.getPropertyValue("font-size"), style.getPropertyValue("font-weight"), style.getPropertyValue("color")];
        """,
        element,
    )
    return font_size, font_weight, text_color


def get_style_properties(element: WebElement, webdriver: Firefox) -> Dict:
    dominant_color = get_dominant_color(element)
    if dominant_color is None:
        return {}
    dominant_color_name = color_db.find_nearest_color(*dominant_color)
    font_size, font_weight, text_color_str = get_font_properties(element, webdriver)
    assert text_color_str.startswith("rgb")
    text_color = (
        literal_eval(text_color_str[4:])
        if text_color_str.startswith("rgba")
        else literal_eval(text_color_str[3:])
    )[:3]
    text_color_name = color_db.find_nearest_color(*text_color)
    return dict(
        dominant_color=(dominant_color_name, dominant_color),
        font_size=font_size,
        font_weight=font_weight,
        text_color=(text_color_name, text_color),
    )


def detect_interface_interference(
    interactive_elements: Dict[str, List],
    banner_selector: str,
    banner_iframe_id: Optional[str],
    webdriver: Firefox,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    accept_label = "Accept"
    reject_labels = [
        "Reject",
        "Save cookie settings",
        "Close/Continue without accepting",
        None,
    ]
    reject_label: str
    if len(interactive_elements[accept_label]) == 0:
        return None, {}
    for label in reject_labels:
        if label is None:
            return None, {}
        if len(interactive_elements[label]) > 0:
            reject_label = label
            break

    styles = []
    for label in [accept_label, reject_label]:
        selector = interactive_elements[label][0][1][0]
        iframe_selector = interactive_elements[label][0][2][0]
        iframe = find_element_by_selector(iframe_selector, webdriver)
        if iframe is not None:
            webdriver.switch_to.frame(iframe)
        element = find_element_by_selector(selector, webdriver)
        if element is None:
            print(
                "Warning: could not find previously detected element for interface interference detection"
            )
            return None, {}
        styles.append(get_style_properties(element, webdriver))
        if iframe is not None:
            webdriver.switch_to.parent_frame()
    accept_style, reject_style = styles
    dominant_colors_match = matching_colors(
        accept_style["dominant_color"][1],
        reject_style["dominant_color"][1],
    )
    text_style_match = (
        matching_colors(accept_style["text_color"][1], reject_style["text_color"][1])
        and float(accept_style["font_size"].replace("px", ""))
        <= float(reject_style["font_size"].replace("px", ""))
        and float(accept_style["font_weight"]) <= float(reject_style["font_weight"])
    )
    banner_iframe = find_element_by_selector(banner_iframe_id, webdriver)
    if banner_iframe is not None:
        webdriver.switch_to.frame(banner_iframe)
    banner = find_element_by_selector(banner_selector, webdriver)
    background_color = get_dominant_color(banner) if banner is not None else None
    if banner_iframe is not None:
        webdriver.switch_to.parent_frame()
    background_color_name = (
        None
        if background_color is None
        else color_db.find_nearest_color(*background_color)
    )
    background_color_obfuscation = (
        background_color is not None
        and not matching_colors(accept_style["dominant_color"][1], background_color)
        and matching_colors(reject_style["dominant_color"][1], background_color)
    )
    interface_interference_found = (
        not dominant_colors_match
        or not text_style_match
        or background_color_obfuscation
    )
    query_summary = dict(
        dominant_colors_match=dominant_colors_match,
        text_style_match=text_style_match,
        background_color_obfuscation=background_color_obfuscation,
        accept_element=dict(label=accept_label, style=accept_style),
        reject_element=dict(label=reject_label, style=reject_style),
        background_color=(background_color_name, background_color),
    )
    return interface_interference_found, query_summary


def detect_forced_action(
    banner_selector: str,
    banner_iframe: Optional[str],
    webdriver: Firefox,
    max_attempts: int = 10,
) -> bool:
    links = get_visible_elements(
        webdriver,
        return_links_only=True,
        element_to_exclude=find_element_by_selector(banner_selector, webdriver)
        if banner_iframe is None
        else None,
    )
    links_ids = get_selector_from_element(links, webdriver)
    random.shuffle(links_ids)
    for num_attempt, link_id in enumerate(links_ids):
        if num_attempt == max_attempts:
            break
        link = find_element_by_selector(link_id, webdriver)
        if link is None or element_is_hidden(link, webdriver):
            continue
        try:
            scroll_into_view(link, webdriver)
            time.sleep(0.2)
            link.click()
            time.sleep(0.2)
            handles = webdriver.window_handles
            if len(handles) > 1:
                # If a new tab is opened, close it
                webdriver.switch_to.window(handles[1])
                webdriver.close()
                webdriver.switch_to.window(handles[0])
            return False
        except (ElementClickInterceptedException, ElementNotInteractableException):
            continue
    return True


def detect_nagging(
    url: str,
    banner_selector: str,
    banner_iframe_id: Optional[str],
    webdriver: Firefox,
    browser_params: BrowserParams,
) -> bool:
    """This function should only be called after interacting with the cookie notice
    and making it disappear"""
    reload_page_and_click_on_elements(webdriver, [], [], url, browser_params)
    banner_iframe = find_element_by_selector(banner_iframe_id, webdriver)
    if banner_iframe is not None:
        webdriver.switch_to.frame(banner_iframe)
    banner = find_element_by_selector(banner_selector, webdriver)
    nagging_detected = banner is not None and not element_is_hidden(banner, webdriver)
    if banner_iframe is not None:
        webdriver.switch_to.parent_frame()
    return nagging_detected
