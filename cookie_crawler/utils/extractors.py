from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import requests
import spacy
from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement

from cookie_crawler.utils.js import (
    extract_text_from_element,
    extract_text_from_interactive_element,
    get_selector_from_element,
)
from cookie_crawler.utils.translate import prepare_for_translation, translate
from database.queries import insert_into_db
from shared_utils import remove_duplicates, strip


@dataclass
class BaseExtractorDataclass:
    """Separating the base dataclass from the base abstract class avoids mypy errors"""

    per_depth_text: List[List[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")


class BaseExtractor(ABC, BaseExtractorDataclass):
    @property
    @abstractmethod
    def table_name(self) -> str:
        pass

    @abstractmethod
    def extract_lines(self, element: WebElement, webdriver: Firefox) -> List[str]:
        pass

    def extract_from_element(
        self, element: WebElement, depth: int, webdriver: Firefox
    ) -> bool:
        prv_size = self.size
        lines = self.extract_lines(element, webdriver)
        if len(lines) == 0:
            return False
        if len(self.per_depth_text) <= depth:
            while len(self.per_depth_text) < depth:
                self.per_depth_text.append([])
            self.per_depth_text.append(remove_duplicates(lines))
        else:
            self.per_depth_text[depth].extend(lines)
            self.per_depth_text[depth] = remove_duplicates(self.per_depth_text[depth])
        return prv_size < self.size

    def extract_from_elements(
        self,
        elements: List[WebElement],
        iframes: List[Optional[WebElement]],
        depth: int,
        webdriver: Firefox,
    ) -> bool:
        new_text_found = False
        for element, iframe in zip(elements, iframes):
            if iframe is not None:
                webdriver.switch_to.frame(iframe)
            new_text_found |= self.extract_from_element(element, depth, webdriver)
            if iframe is not None:
                webdriver.switch_to.parent_frame()
        return new_text_found

    def prune(self) -> None:
        if len(self.per_depth_text) == 0:
            return
        all_text = self.per_depth_text[0].copy()
        for depth in range(1, len(self.per_depth_text)):
            self.per_depth_text[depth] = [
                sent for sent in self.per_depth_text[depth] if sent not in all_text
            ]
            all_text.extend(self.per_depth_text[depth])

    def print(self) -> None:
        for depth, lines in enumerate(self.per_depth_text):
            print(f"\nSentences at depth {depth}: {lines}")

    @property
    def size(self) -> int:
        return sum([len(t) for t in self.per_depth_text])

    def insert_into_db(self, website: Dict) -> None:
        self.prune()
        records = [
            dict(
                website_id=website["id"],
                num_clicks=depth,
                text=line,
            )
            for depth, lines in enumerate(self.per_depth_text)
            for line in lines
        ]
        insert_into_db(self.table_name, records)


@dataclass
class TextExtractor(BaseExtractor):
    translate_non_english_text: bool = True
    source_language: Optional[str] = None

    @property
    def table_name(self) -> str:
        return "cb_text"

    def extract_lines(self, element: WebElement, webdriver: Firefox) -> List[str]:
        text = extract_text_from_element(element, webdriver)
        sents: List[str] = []
        if self.translate_non_english_text:
            try:
                text = translate(text, self.source_language, verbose=False)
            except:
                pass
        for line in text.split("\n"):
            if strip(line) == "":
                continue
            sents.extend(map(str, self.nlp(strip(line)).sents))
        return sents

    def find_text(self, text: str, depth: Optional[int] = None) -> bool:
        if depth is None:
            depths = list(range(len(self.per_depth_text)))
        else:
            depths = [depth]

        mention_found = False
        for depth in depths:
            mention_found |= any(text in s for s in self.per_depth_text[depth])
        return mention_found

    def get_li_mentions(self) -> Dict[str, bool]:
        crawl_results: Dict[str, bool] = {}
        crawl_results["mentions_legitimate_interest_in_initial_text"] = self.find_text(
            "legitimate interest", 0
        )
        crawl_results["mentions_legitimate_interest"] = self.find_text(
            "legitimate interest"
        )
        return crawl_results


@dataclass
class LinkExtractor(BaseExtractor):
    @property
    def table_name(self) -> str:
        return "cb_links"

    def extract_lines(self, element: WebElement, webdriver: Firefox) -> List[str]:
        links = element.find_elements("tag name", "a")
        hrefs = [link.get_attribute("href") for link in links]
        hrefs = [href for href in hrefs if href is not None]
        return hrefs


@dataclass
class InteractiveElementsTextExtractor(BaseExtractorDataclass):
    elements_with_assigned_text: Set[str] = field(default_factory=set)
    per_depth_selectors: List[List[str]] = field(default_factory=list)
    translate_non_english_text: bool = True
    source_language: Optional[str] = None
    ietc_model_url: Optional[str] = None

    @property
    def table_name(self) -> str:
        return "interactive_elements_text"

    def assign_text_to_ui_element(
        self,
        element: WebElement,
        webdriver: Firefox,
        selector: Optional[str] = None,
    ) -> str:
        if selector is None:
            selector = get_selector_from_element(element, webdriver)

        if selector is None or selector in self.elements_with_assigned_text:
            return ""

        cur_text = extract_text_from_interactive_element(element, webdriver)
        if cur_text != "":
            self.elements_with_assigned_text.add(selector)
            return cur_text
        return ""

    def extract_from_element(
        self,
        element: WebElement,
        selector: str,
        iframe_id: Optional[str],
        depth: int,
        webdriver: Firefox,
        extracted_text: List[str],
        corresponding_selectors: List[str],
        corresponding_iframes: List[Optional[str]],
    ) -> None:
        if selector in self.elements_with_assigned_text:
            # The text corresponding to a certain element may change after user interaction
            # (e.g. a "Show details" that changes to "Hide details" when a user clicks
            # on it). This approach does not take this into account (i.e. we would only
            # assign Show details to such a button).
            return
        text = self.assign_text_to_ui_element(element, webdriver, selector).strip()
        self.elements_with_assigned_text.add(selector)
        if text != "":
            extracted_text.append(text)
            corresponding_selectors.append(selector)
            corresponding_iframes.append(iframe_id)

    def extract_from_elements(
        self,
        elements: List[WebElement],
        iframes: List[Optional[WebElement]],
        depth: int,
        webdriver: Firefox,
        selectors: Optional[List[str]] = None,
        make_predictions: bool = False,
    ) -> Optional[Dict[str, List]]:
        if selectors is None:
            selectors, elements = get_selector_from_element(
                elements, webdriver, return_matched_elements=True
            )
        text: List[str] = []
        corresponding_selectors: List[str] = []
        corresponding_iframes: List[Optional[str]] = []
        for element, selector, iframe in zip(elements, selectors, iframes):
            iframe_id = (
                None if iframe is None else get_selector_from_element(iframe, webdriver)
            )
            if iframe is not None:
                webdriver.switch_to.frame(iframe)
            self.extract_from_element(
                element,
                selector,
                iframe_id,
                depth,
                webdriver,
                text,
                corresponding_selectors,
                corresponding_iframes,
            )
            if iframe is not None:
                webdriver.switch_to.parent_frame()

        while len(self.per_depth_text) <= depth:
            self.per_depth_text.append([])
            self.per_depth_selectors.append([])

        map_translated_to_original: Dict[str, str] = {}
        original_text = text.copy()
        if self.translate_non_english_text:
            try:
                SEP_TOKEN = "\n\n"
                text = translate(
                    prepare_for_translation(SEP_TOKEN.join(text)),
                    self.source_language,
                    verbose=False,
                ).split(SEP_TOKEN)
            except:
                pass

        if len(original_text) == len(text):
            for sent_original, sent_translated in zip(original_text, text):
                map_translated_to_original[sent_translated] = sent_original
        self.per_depth_text[depth].extend(text)
        self.per_depth_selectors[depth].extend(corresponding_selectors)
        IECT_LABELS = [
            "Accept",
            "Reject",
            "Cookies settings",
            "Save cookie settings",
            "Close/Continue without accepting",
        ]
        structured_preds: Optional[Dict[str, List]] = None
        if make_predictions:
            assert self.ietc_model_url is not None
            structured_preds = {label: [] for label in IECT_LABELS}
            request = requests.post(url=self.ietc_model_url, data={"s": text})
            try:
                preds = requests.post(url=self.ietc_model_url, data={"s": text}).json()
            except:
                raise ValueError(f"Could not query the IETC model: {request.content}")
            for pred, sent, selector, iframe_id in zip(
                preds, text, corresponding_selectors, corresponding_iframes
            ):
                if pred in IECT_LABELS:
                    structured_preds[pred].append((sent, selector, iframe_id))
        return structured_preds

    def insert_into_db(self, website: Dict) -> None:
        for depth in range(len(self.per_depth_text)):
            text, selectors = [], []
            for line, selector in zip(
                self.per_depth_text[depth], self.per_depth_selectors[depth]
            ):
                if line in text:
                    continue
                text.append(line)
                selectors.append(selector)
            self.per_depth_text[depth] = text
            self.per_depth_selectors[depth] = selectors
        records = [
            dict(
                website_id=website["id"],
                num_clicks=depth,
                text=text,
                selector=selector,
            )
            for depth in range(len(self.per_depth_text))
            for text, selector in zip(
                self.per_depth_text[depth], self.per_depth_selectors[depth]
            )
        ]
        insert_into_db(self.table_name, records)

    def print(self) -> None:
        for depth, lines in enumerate(self.per_depth_text):
            print(f"\nSentences at depth {depth}: {lines}")
