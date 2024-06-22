import time
from typing import List, Optional, Tuple, Union

from selenium.common.exceptions import (
    JavascriptException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from cookie_crawler.utils.translate import translate
from shared_utils import repeat


@repeat()
def contains(a: WebElement, b: WebElement, webdriver: Firefox) -> bool:
    return webdriver.execute_script("return arguments[0].contains(arguments[1])", a, b)


def scroll_into_view(element: WebElement, webdriver: Firefox) -> None:
    webdriver.execute_script(
        'arguments[0].scrollIntoView({block: "center", inline: "nearest"});', element
    )


@repeat()
def click(element: WebElement, webdriver: Firefox, sleep: int = 3) -> None:
    webdriver.execute_script(
        "arguments[0].scrollIntoView(true);arguments[0].click();", element
    )
    time.sleep(sleep)


@repeat()
def get_selector_from_element(
    elements: Union[WebElement, List[WebElement]],
    webdriver: Firefox,
    keep_text_elements_only: bool = False,
    return_matched_elements: bool = False,
) -> Optional[Union[List[str], str, Tuple[List[str], List[WebElement]]]]:
    unique_element = False
    if isinstance(elements, WebElement):
        elements = [elements]
        unique_element = True
    selectors, matched_elements = webdriver.execute_script(
        open("cookie_crawler/scripts/optimal_select.js").read(),
        elements,
        keep_text_elements_only,
    )
    if unique_element:
        if len(selectors) > 0:
            return selectors[0]
        raise RuntimeError("Unable to assign selector to element.")
    else:
        if return_matched_elements:
            return selectors, matched_elements
        return selectors


def clear_data(webdriver: Firefox) -> None:
    webdriver.get("about:preferences#privacy")
    webdriver.execute_script(open("cookie_crawler/scripts/clear_data.js").read())
    WebDriverWait(webdriver, 20).until(EC.alert_is_present())
    webdriver.switch_to.alert.accept()


def element_is_hidden(element: WebElement, webdriver: Firefox) -> bool:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/element_is_hidden.js").read(),
        element,
    )


def extract_text_from_element(
    element: WebElement, webdriver: Firefox, exclude_links: bool = False
) -> str:
    text = webdriver.execute_script(
        open("cookie_crawler/scripts/extract_text_from_element.js").read(),
        element,
        exclude_links,
    )
    return text


def get_link_to_text_ratio(element: WebElement, webdriver: Firefox) -> float:
    full_text = extract_text_from_element(element, webdriver, exclude_links=False)
    no_link_text = extract_text_from_element(element, webdriver, exclude_links=True)
    return 1.0 - len(no_link_text) / len(full_text) if len(full_text) > 0 else 0.0


@repeat(raise_exception_if_fails=False, sleep=3)
def find_elements_by_selectors(
    selectors: List[str], webdriver: Firefox
) -> Tuple[List[WebElement], List[str]]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/find_elements_by_selector.js").read(),
        selectors,
    )


def find_element_by_selector(
    selector: Optional[str],
    webdriver: Firefox,
) -> Optional[WebElement]:
    if selector is None:
        return None
    elements, _ = find_elements_by_selectors([selector], webdriver)
    if len(elements) == 0:
        return None
    return elements[0]


@repeat()
def get_z_index(
    element: WebElement, webdriver: Firefox, default_value: Optional[int] = 0
) -> Optional[int]:
    try:
        return webdriver.execute_script(
            open("cookie_crawler/scripts/get_z_index.js").read(), element
        )
    except JavascriptException:
        return default_value


@repeat()
def find_elements_with_geq_z_index(
    z_index: int, webdriver: Firefox
) -> List[WebElement]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/find_elements_with_geq_z_index.js").read(),
        z_index,
    )


@repeat(raise_exception_if_fails=False)
def get_active_element(webdriver: Firefox) -> WebElement:
    active_element = webdriver.switch_to.active_element
    if active_element is None:
        raise WebDriverException("Could not switch to active element.")
    return active_element


@repeat()
def get_visible_elements(
    webdriver: Firefox,
    return_links_only: bool = False,
    element_to_exclude: Optional[WebElement] = None,
) -> List[WebElement]:
    elements = webdriver.execute_script(
        open("cookie_crawler/scripts/get_visible_elements.js").read(),
        return_links_only,
        element_to_exclude,
    )
    return elements


def get_visible_elements_ids(
    webdriver: Firefox,
    keep_text_elements_only: bool = True,
) -> List[str]:
    elements = get_visible_elements(webdriver)
    selectors = get_selector_from_element(elements, webdriver, keep_text_elements_only)
    assert isinstance(selectors, list)
    return selectors


def is_stale(element: WebElement) -> bool:
    try:
        _ = element.text
        return False
    except StaleElementReferenceException:
        return True


def scroll_to_bottom(webdriver: Firefox) -> None:
    try:
        webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    except WebDriverException:
        pass


@repeat(num_tries=2, raise_exception_if_fails=False, return_value_if_fails=[])
def get_iframes(webdriver: Firefox) -> List[WebElement]:
    return webdriver.find_elements("tag name", "iframe")


@repeat()
def get_parent(element: WebElement, webdriver: Firefox) -> WebElement:
    return webdriver.execute_script("return arguments[0].parentNode", element)


@repeat(num_tries=1)
def get_neighbors(
    element: WebElement, webdriver: Firefox
) -> Tuple[List[WebElement], List[str]]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/get_neighbors.js").read(), element
    )


@repeat()
def extract_text_from_interactive_element(
    element: WebElement,
    webdriver: Firefox,
    source_language: Optional[str] = None,
    translate_non_english_text: bool = False,
) -> str:
    text = extract_text_from_element(element, webdriver)

    if "\n" in text.strip():
        text = ""

    for attribute in ["aria-label", "value"]:
        if text.strip() == "":
            text = webdriver.execute_script(
                open("cookie_crawler/scripts/get_attribute.js").read(),
                element,
                attribute,
            )
            try:
                if text != "" and translate_non_english_text:
                    text = translate(text, source_language, verbose=False)
            except:
                pass
    return text


@repeat()
def extract_clickable_elements(
    element: WebElement, webdriver: Firefox
) -> List[WebElement]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/extract_clickable_elements.js").read(),
        element,
    )


@repeat()
def get_first_and_last_elements(webdriver: Firefox) -> List[WebElement]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/get_first_and_last_elements.js").read(),
    )


@repeat()
def sort_elements(elements: List[WebElement], webdriver: Firefox) -> List[WebElement]:
    return webdriver.execute_script(
        open("cookie_crawler/scripts/sort_elements.js").read(),
        elements,
    )
