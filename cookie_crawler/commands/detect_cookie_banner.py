import time
from typing import Dict, List, Optional, Set, Tuple

import langdetect
import spacy
from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement

from cookie_crawler.commands.get_command import is_stale
from cookie_crawler.utils.js import (
    extract_clickable_elements,
    extract_text_from_element,
    find_element_by_selector,
    find_elements_by_selectors,
    find_elements_with_geq_z_index,
    get_first_and_last_elements,
    get_iframes,
    get_selector_from_element,
)
from shared_utils import repeat


def find_cookie_mention(text: str, language_params: Dict) -> bool:
    model_type = language_params["model"]["type"]
    model_name = language_params["model"]["name"]
    cookie_keywords = language_params["keywords"]["cookie"]
    if model_type == "spacy":
        nlp = spacy.load(model_name)
        lines = text.split("\n")
        for line in lines:
            for sentence in nlp(line.strip()).sents:
                if all(
                    keyword.lower() not in str(sentence).lower()
                    for keyword in cookie_keywords
                ):
                    continue
                for token in sentence:
                    if token.pos_ == "VERB" or token.pos_ == "AUX":
                        return True
    else:
        raise ValueError(f"Unsupported model type for notice detection {model_type}")
    return False


def filter_detected_banner_elements(
    elements: List[WebElement],
    webdriver: Firefox,
    source_language: str,
    supported_languages: Dict,
    black_list_ids: Optional[Set[str]] = None,
    depth: int = 0,
) -> Tuple[List[WebElement], List[str]]:
    if len(elements) == 0:
        return elements, []

    filtered_elements: List[WebElement] = []
    filtered_selectors: List[str] = []

    selectors, elements = get_selector_from_element(
        elements, webdriver, return_matched_elements=True
    )

    for element, selector in zip(elements, selectors):
        if (
            is_stale(element)
            or (black_list_ids is not None and selector in black_list_ids)
            or (depth == 0 and len(extract_clickable_elements(element, webdriver)) > 10)
        ):
            continue

        element_text = extract_text_from_element(element, webdriver)
        if len(element_text.strip()) == 0:
            continue

        if source_language not in supported_languages or not find_cookie_mention(
            element_text, supported_languages[source_language]
        ):
            text_language = None
            try:
                text_language = langdetect.detect(element_text)
            except:
                pass
            if (
                text_language == source_language
                or text_language not in supported_languages
                or not find_cookie_mention(
                    element_text, supported_languages[text_language]
                )
            ):
                continue

        filtered_elements.append(element)
        filtered_selectors.append(selector)
        break

    return webdriver.execute_script(
        open("cookie_crawler/scripts/filter_banner_elements.js").read(),
        filtered_elements,
        filtered_selectors,
    )


@repeat()
def detect_cookie_banner(
    webdriver: Firefox,
    source_language: str,
    supported_languages: Dict,
    use_z_index: bool,
    explore_iframes: bool,
    banner_selector: Optional[str] = None,
    candidate_selectors: Optional[List[str]] = None,
    black_list_ids: Optional[Set[str]] = None,
    depth: int = 0,
) -> Tuple[List[WebElement], List[str], List[Optional[WebElement]], bool]:
    iframes = [None]
    if explore_iframes:
        iframes.extend(
            [iframe for iframe in get_iframes(webdriver) if not is_stale(iframe)]
        )
    elements: List[WebElement] = []
    elements_ids: List[str] = []
    iframes_corresponding_to_elements: List[Optional[WebElement]] = []
    z_index_detection = False
    min_z_index = 1
    for iframe in iframes:
        if iframe is not None:
            try:
                webdriver.switch_to.frame(iframe)
            except:
                continue
            min_z_index = 0

        if candidate_selectors is not None:
            tmp_elements, _ = find_elements_by_selectors(candidate_selectors, webdriver)
            tmp_elements, tmp_selectors = filter_detected_banner_elements(
                tmp_elements,
                webdriver,
                source_language,
                supported_languages,
                black_list_ids=black_list_ids,
                depth=depth,
            )
            elements.extend(tmp_elements)
            elements_ids.extend(tmp_selectors)
            iframes_corresponding_to_elements.extend([iframe] * len(tmp_elements))

        if len(elements) == 0 and use_z_index:
            tmp_elements = find_elements_with_geq_z_index(min_z_index, webdriver)
            tmp_elements = [e for e in tmp_elements if e not in elements]
            tmp_elements, tmp_selectors = filter_detected_banner_elements(
                tmp_elements,
                webdriver,
                source_language,
                supported_languages,
                black_list_ids=black_list_ids,
                depth=depth,
            )
            if len(elements) == 0 and len(tmp_elements) > 0:
                z_index_detection = True
            elements.extend(tmp_elements)
            elements_ids.extend(tmp_selectors)
            iframes_corresponding_to_elements.extend([iframe] * len(tmp_elements))

        if len(elements) == 0:
            tmp_elements = get_first_and_last_elements(webdriver)
            tmp_elements, tmp_selectors = filter_detected_banner_elements(
                tmp_elements,
                webdriver,
                source_language,
                supported_languages,
                black_list_ids=black_list_ids,
                depth=0,
            )
            elements.extend(tmp_elements)
            elements_ids.extend(tmp_selectors)
            iframes_corresponding_to_elements.extend([iframe] * len(tmp_elements))

        if banner_selector is not None:
            banner = find_element_by_selector(banner_selector, webdriver)
            if banner is not None and banner not in elements:
                elements.append(banner)
                elements_ids.append(banner_selector)
                iframes_corresponding_to_elements.append(iframe)

        time.sleep(1)

        if iframe is not None:
            webdriver.switch_to.parent_frame()

    return elements, elements_ids, iframes_corresponding_to_elements, z_index_detection
