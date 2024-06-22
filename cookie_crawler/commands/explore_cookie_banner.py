import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openwpm.config import BrowserParams

from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement

from cookie_crawler.commands.detect_cookie_banner import detect_cookie_banner
from cookie_crawler.commands.detect_dark_patterns import (
    detect_interface_interference,
    detect_nagging,
)
from cookie_crawler.commands.get_command import (
    browse,
    is_stale,
    load_page,
    reload_page_and_click_on_elements,
)
from cookie_crawler.utils.domains import is_url_to_different_domain
from cookie_crawler.utils.extractors import (
    InteractiveElementsTextExtractor,
    LinkExtractor,
    TextExtractor,
)
from cookie_crawler.utils.js import (
    click,
    element_is_hidden,
    extract_clickable_elements,
    find_element_by_selector,
    get_selector_from_element,
    get_visible_elements_ids,
    scroll_to_bottom,
    sort_elements,
)
from database.queries import insert_into_db
from shared_utils import get_timestamp, repeat


@repeat(
    raise_exception_if_fails=True, return_value_if_fails=([], [], [], [], []), sleep=0
)
def get_interactive_elements(
    webdriver: Firefox,
    parents: List[WebElement],
    parents_iframes: List[Optional[WebElement]],
    depth: int = 0,
) -> Tuple[
    List[str],
    List[str],
    List[WebElement],
    List[Optional[WebElement]],
    List[Optional[str]],
]:
    elements = []
    selectors = []
    hrefs = []
    corresponding_iframes = []
    corresponding_iframes_ids = []
    for parent, iframe in zip(parents, parents_iframes):
        iframe_id = (
            None if iframe is None else get_selector_from_element(iframe, webdriver)
        )
        if iframe is not None:
            webdriver.switch_to.frame(iframe)
        tmp_elements = extract_clickable_elements(parent, webdriver)
        tmp_elements = list(reversed(sort_elements(tmp_elements, webdriver)))

        if depth > 0 and len(tmp_elements) > 50:
            tmp_elements = tmp_elements[:25] + tmp_elements[-25:]

        try:
            tmp_selectors, tmp_elements = get_selector_from_element(
                tmp_elements, webdriver, return_matched_elements=True
            )
            tmp_hrefs = [element.get_attribute("href") for element in tmp_elements]
        except:
            tmp_selectors, tmp_hrefs, tmp_elements = [], [], []

        elements.extend(tmp_elements)
        selectors.extend(tmp_selectors)
        hrefs.extend(tmp_hrefs)
        corresponding_iframes.extend([iframe] * len(tmp_elements))
        corresponding_iframes_ids.extend([iframe_id] * len(tmp_elements))

        if iframe is not None:
            webdriver.switch_to.parent_frame()

    return selectors, hrefs, elements, corresponding_iframes, corresponding_iframes_ids


@dataclass
class Node:
    id: str
    iframe_id: Optional[str]
    href: Optional[str]
    last_child: bool
    parents: List[str] = field(default_factory=list)
    parents_iframes: List[Optional[str]] = field(default_factory=list)
    parent: Optional["Node"] = None

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parents = self.parent.parents.copy()
            self.parents.append(self.parent.id)
            self.parents_iframes = self.parent.parents_iframes.copy()
            self.parents_iframes.append(self.parent.iframe_id)
            self.parent = None

    @property
    def depth(self) -> int:
        return len(self.parents) + 1


# flake8: noqa: C901
def explore_cookie_banner_naive(
    banner_selector: str,
    banner_iframe_id: Optional[str],
    website: Dict,
    webdriver: Firefox,
    browser_params: BrowserParams,
    config: Dict,
    max_depth: int,
    load_timestamp: datetime,
) -> Dict[str, Union[int, bool]]:
    crawl_results: Dict[str, Union[int, bool]] = dict()
    depth_0_elements = set(get_visible_elements_ids(webdriver))
    start_time = time.time()

    iframe = (
        None
        if banner_iframe_id is None
        else find_element_by_selector(banner_iframe_id, webdriver)
    )
    if iframe is not None:
        webdriver.switch_to.frame(iframe)
    banner = find_element_by_selector(banner_selector, webdriver)

    if banner is None:
        raise ValueError(
            "Cookie banner not found. It is likely that the webpage did not reload properly"
        )
    (
        root_elements_ids,
        root_elements_hrefs,
        root_elements,
        _,
        _,
    ) = get_interactive_elements(webdriver, [banner], [None])
    translate_non_english_text = browser_params.custom_params[
        "translate_non_english_text"
    ]
    text_extractor = TextExtractor(
        source_language=website["language"],
        translate_non_english_text=translate_non_english_text,
    )
    text_extractor.extract_from_element(banner, 0, webdriver)
    link_extractor = LinkExtractor()
    link_extractor.extract_from_element(banner, 0, webdriver)
    if config["extract_interactive_elements_text"]:
        ie_text_extractor = InteractiveElementsTextExtractor(
            source_language=website["language"],
            translate_non_english_text=translate_non_english_text,
        )
        ie_text_extractor.extract_from_elements(
            root_elements, [None] * len(root_elements), 0, webdriver, root_elements_ids
        )
    else:
        ie_text_extractor = None
    elements_not_yielding_new_text: Set[str] = set()

    crawl_results["interaction_depth"] = 0
    num_interactive_elements_per_depth: Dict[int, int] = dict()
    if len(root_elements) > 0:
        num_interactive_elements_per_depth[0] = len(root_elements)

    if iframe is not None:
        webdriver.switch_to.parent_frame()

    for root_num, (root_id, root_href) in enumerate(
        zip(root_elements_ids, root_elements_hrefs)
    ):
        stack = [Node(root_id, banner_iframe_id, root_href, root_num == 0)]
        while len(stack) > 0 and (time.time() - start_time < config["timeout"] / 2.5):
            cur_node = stack.pop()
            if (
                cur_node.depth > max_depth
                or (
                    is_url_to_different_domain(cur_node.href, website["url"])
                    and is_url_to_different_domain(cur_node.href, webdriver.current_url)
                )
                or (
                    browser_params.custom_params[
                        "explore_elements_yielding_new_text_only"
                    ]
                    and cur_node.id in elements_not_yielding_new_text
                )
            ):
                continue

            if not cur_node.last_child:
                load_timestamp, click_timestamp = reload_page_and_click_on_elements(
                    webdriver,
                    cur_node.parents,
                    cur_node.parents_iframes,
                    website["url"],
                    browser_params,
                    delete_cookies=True,
                )[0]

            cur_iframe = find_element_by_selector(cur_node.iframe_id, webdriver)
            body = webdriver.find_element("tag name", "body")
            if cur_iframe is not None:
                webdriver.switch_to.frame(cur_iframe)

            cur_element = find_element_by_selector(cur_node.id, webdriver)
            if cur_element is None:
                if cur_iframe is not None:
                    webdriver.switch_to.parent_frame()
                continue

            click(cur_element, webdriver)

            print(
                " " * 4 * (cur_node.depth - 1),
                f"Clicked on element {cur_node.id}.",
                sep="",
            )
            crawl_results["interaction_depth"] = max(
                crawl_results["interaction_depth"], cur_node.depth
            )

            if cur_iframe is not None:
                webdriver.switch_to.parent_frame()

            if is_stale(body) or len(webdriver.window_handles) > 1:
                continue

            banner_windows, _, banner_windows_iframes, _ = detect_cookie_banner(
                webdriver,
                website["language"],
                config["supported_languages"],
                use_z_index=browser_params.custom_params[
                    "extract_cookie_settings_with_z_index"
                ],
                explore_iframes=True,
                black_list_ids=depth_0_elements,
                banner_selector=banner_selector,
                depth=1,
            )

            if time.time() - start_time > config["timeout"] / 2.5:
                break

            if browser_params.custom_params["extract_cookie_banner_text"]:
                found_new_text = text_extractor.extract_from_elements(
                    banner_windows, banner_windows_iframes, cur_node.depth, webdriver
                )
                if not found_new_text:
                    elements_not_yielding_new_text.add(cur_node.id)
                link_extractor.extract_from_elements(
                    banner_windows, banner_windows_iframes, cur_node.depth, webdriver
                )

            (
                children_ids,
                children_hrefs,
                children_elements,
                children_elements_iframes,
                children_elements_iframes_ids,
            ) = get_interactive_elements(
                webdriver,
                banner_windows,
                banner_windows_iframes,
                depth=1,
            )

            if time.time() - start_time > config["timeout"] / 2.5:
                break

            if len(children_elements) > 0:
                num_interactive_elements_per_depth[cur_node.depth] = len(
                    children_elements
                ) + num_interactive_elements_per_depth.get(cur_node.depth, 0)

            if len(children_elements) > 50:
                # This usually happens when the cookie banner displays a long list of
                # trackers. The command is likely to timeout in this case.
                # We reduce the number of children to visit to include the first 5 and
                # the last 5.
                children_elements = children_elements[:25] + children_elements[-25:]
                children_ids = children_ids[:25] + children_ids[-25:]
                children_hrefs = children_hrefs[:25] + children_hrefs[-25:]
                children_elements_iframes = (
                    children_elements_iframes[:25] + children_elements_iframes[-25:]
                )
                children_elements_iframes_ids = (
                    children_elements_iframes_ids[:25]
                    + children_elements_iframes_ids[-25:]
                )

            if ie_text_extractor is not None:
                ie_text_extractor.extract_from_elements(
                    children_elements,
                    children_elements_iframes,
                    cur_node.depth,
                    webdriver,
                    children_ids,
                )

            for e_num, (e_id, e_href, e_iframe_id) in enumerate(
                zip(children_ids, children_hrefs, children_elements_iframes_ids)
            ):
                if e_id != cur_node.id:
                    stack.append(
                        Node(
                            e_id,
                            e_iframe_id,
                            e_href,
                            e_num == len(children_ids) - 1,
                            parent=cur_node,
                        )
                    )

    if len(num_interactive_elements_per_depth) > 0:
        insert_into_db(
            "num_interactive_elements",
            [
                dict(
                    website_id=website["id"],
                    depth=depth,
                    num_elements=num_elements,
                )
                for depth, num_elements in num_interactive_elements_per_depth.items()
            ],
        )

    if browser_params.custom_params["extract_cookie_banner_text"]:
        print(get_timestamp(website["name"]), f"Finished text extraction.")
        text_extractor.insert_into_db(website)
        text_extractor.print()
        link_extractor.insert_into_db(website)
        link_extractor.print()
        if ie_text_extractor is not None:
            ie_text_extractor.insert_into_db(website)
            ie_text_extractor.print()
        crawl_results.update(text_extractor.get_li_mentions())

    return crawl_results


def makes_banner_disappear(
    webdriver: Firefox,
    elements_ids: List[str],
    iframes_ids: List[str],
    url: str,
    banner_id: str,
    browser_params: BrowserParams,
) -> bool:
    try:
        _, loads_new_page = reload_page_and_click_on_elements(
            webdriver,
            elements_ids,
            iframes_ids,
            url,
            browser_params,
            delete_cookies=True,
        )
    except:
        return False
    iframe = (
        None
        if iframes_ids[0] is None
        else find_element_by_selector(iframes_ids[0], webdriver)
    )
    if iframe is not None:
        webdriver.switch_to.frame(iframe)
    banner = find_element_by_selector(banner_id, webdriver)
    banner_not_found = banner is None or element_is_hidden(banner, webdriver)
    if iframe is not None:
        webdriver.switch_to.parent_frame()
    return banner_not_found


LABELS = {
    "accept": "Accept",
    "reject": "Reject",
    "save": "Save cookie settings",
    "close": "Close/Continue without accepting",
    "settings": "Cookies settings",
}


def explore_cookie_banner_with_ietc_model(
    banner_selector: str,
    banner_iframe_id: Optional[str],
    website: Dict,
    webdriver: Firefox,
    browser_params: BrowserParams,
    experiment_config: Dict,
) -> Dict[str, Optional[Union[int, bool, Dict[str, Any]]]]:
    crawl_results: Dict[str, Optional[Union[int, bool, Dict[str, Any]]]] = dict()
    depth_0_elements = set(get_visible_elements_ids(webdriver))

    iframe = (
        None
        if banner_iframe_id is None
        else find_element_by_selector(banner_iframe_id, webdriver)
    )
    if iframe is not None:
        webdriver.switch_to.frame(iframe)
    banner = find_element_by_selector(banner_selector, webdriver)

    if banner is None:
        raise ValueError(
            "Cookie banner not found. Either the page did not reload properly, "
            "or some selectors changed after reloading the page."
        )
    (
        root_elements_ids,
        _,
        root_elements,
        _,
        _,
    ) = get_interactive_elements(webdriver, [banner], [None])

    ie_text_extractor = InteractiveElementsTextExtractor(
        source_language=website["language"],
        translate_non_english_text=browser_params.custom_params[
            "translate_non_english_text"
        ],
        ietc_model_url=browser_params.custom_params["ietc_model_url"],
    )
    if iframe is not None:
        webdriver.switch_to.parent_frame()
    filtered_interactive_elements: Dict[str, List] = {}
    interactive_elements = ie_text_extractor.extract_from_elements(
        root_elements,
        [iframe] * len(root_elements),
        0,
        webdriver,
        root_elements_ids,
        make_predictions=True,
    )
    assert interactive_elements is not None

    visited_elements: Set[Tuple] = set()
    nagging_detection_executed = False
    for label in interactive_elements:
        if label != LABELS["settings"]:
            filtered_interactive_elements[label] = []
            for text, selector, iframe_id in interactive_elements[label]:
                visited_elements.add((text, selector, iframe_id))
                if makes_banner_disappear(
                    webdriver,
                    [selector],
                    [iframe_id],
                    website["url"],
                    banner_selector,
                    browser_params,
                ):
                    filtered_interactive_elements[label].append(
                        ([text], [selector], [iframe_id])
                    )
                    print(
                        get_timestamp(website["name"]),
                        "Found option that makes banner disappear: ",
                        text,
                    )
                    if (
                        not nagging_detection_executed
                        and label != LABELS["accept"]
                        and browser_params.custom_params["detect_dark_patterns"]
                    ):
                        webdriver.save_screenshot(
                            os.path.join(
                                website["save_path"],
                                "nagging_detection_before_reload.png",
                            )
                        )
                        crawl_results.update(
                            nagging_detected=detect_nagging(
                                website["url"],
                                banner_selector,
                                banner_iframe_id,
                                webdriver,
                                browser_params,
                            )
                        )
                        webdriver.save_screenshot(
                            os.path.join(
                                website["save_path"],
                                "nagging_detection_after_reload.png",
                            )
                        )
                        nagging_detection_executed = True

    if browser_params.custom_params["detect_dark_patterns"]:
        _, _ = load_page(website["url"], webdriver, browser_params, delete_cookies=True)
        (
            interface_interference_detected,
            interface_interference_summary,
        ) = detect_interface_interference(
            filtered_interactive_elements,
            banner_selector,
            banner_iframe_id,
            webdriver,
        )
        crawl_results.update(
            interface_interference_detected=interface_interference_detected,
            interface_interference_analysis=interface_interference_summary,
        )

    for text, selector, iframe_id in interactive_elements[LABELS["settings"]]:
        try:
            _, loads_new_page = reload_page_and_click_on_elements(
                webdriver,
                [selector],
                [iframe_id],
                website["url"],
                browser_params,
                delete_cookies=True,
            )
            if loads_new_page:
                continue
        except:
            continue
        banner_windows, _, banner_windows_iframes, _ = detect_cookie_banner(
            webdriver,
            website["language"],
            experiment_config["supported_languages"],
            use_z_index=browser_params.custom_params[
                "extract_cookie_settings_with_z_index"
            ],
            explore_iframes=True,
            black_list_ids=depth_0_elements,
            banner_selector=banner_selector,
            depth=1,
        )
        (
            children_ids,
            _,
            children_elements,
            children_elements_iframes,
            _,
        ) = get_interactive_elements(
            webdriver,
            banner_windows,
            banner_windows_iframes,
            depth=1,
        )
        if len(children_elements) > 50:
            children_elements = children_elements[:25] + children_elements[-25:]
            children_ids = children_ids[:25] + children_ids[-25:]
            children_elements_iframes = (
                children_elements_iframes[:25] + children_elements_iframes[-25:]
            )
        tmp_interactive_elements = ie_text_extractor.extract_from_elements(
            children_elements,
            children_elements_iframes,
            1,
            webdriver,
            selectors=children_ids,
            make_predictions=True,
        )
        assert tmp_interactive_elements is not None

        for label in tmp_interactive_elements:
            if label != LABELS["settings"]:
                for text_2, selector_2, iframe_id_2 in tmp_interactive_elements[label]:
                    if (text_2, selector_2, iframe_id_2) in visited_elements:
                        continue
                    visited_elements.add((text_2, selector_2, iframe_id_2))
                    if makes_banner_disappear(
                        webdriver,
                        [selector, selector_2],
                        [iframe_id, iframe_id_2],
                        website["url"],
                        banner_selector,
                        browser_params,
                    ):
                        print(
                            get_timestamp(website["name"]),
                            f"Found option that makes banner disappear: {text} -> {text_2}",
                        )
                        filtered_interactive_elements[label].append(
                            (
                                [text, text_2],
                                [selector, selector_2],
                                [iframe_id, iframe_id_2],
                            )
                        )

    print(get_timestamp(website["name"]), "Collecting cookies for consent options.")
    for label_abbrv, label in LABELS.items():
        if label_abbrv == "settings":
            continue
        if label_abbrv not in filtered_interactive_elements:
            crawl_results[f"{label_abbrv}_button_detected"] = 0
        for idx, (texts, selectors, iframes_ids) in enumerate(
            filtered_interactive_elements[label]
        ):
            if idx == 0:
                crawl_results[f"{label_abbrv}_button_detected"] = len(texts)
                load_timestamp, click_timestamp = reload_page_and_click_on_elements(
                    webdriver,
                    selectors,
                    iframes_ids,
                    website["url"],
                    browser_params,
                    delete_cookies=True,
                )[0]
                scroll_to_bottom(webdriver)
                browse(
                    website["url"],
                    webdriver,
                    num_links=browser_params.custom_params["num_links_for_browsing"],
                    sleep=1,
                    seed=42,
                )
                print(
                    get_timestamp(website["name"]),
                    f"Browsed website after clicking on {texts} ({label})",
                )
                time.sleep(1)
                end_timestamp = datetime.utcnow()
                insert_into_db(
                    "cookie_timestamps",
                    dict(
                        website_id=website["id"],
                        collection_strategy=label,
                        visit_id=website["visit_id"],
                        start_timestamp=load_timestamp,
                        end_timestamp=end_timestamp,
                        num_cookies=len(webdriver.get_cookies()),
                        click_timestamp=click_timestamp,
                    ),
                )
            insert_into_db(
                "consent_options",
                dict(
                    website_id=website["id"],
                    consent_option=label,
                    text="\n".join(texts),
                    selector="\n".join(selectors),
                    iframe="\n".join(map(str, iframes_ids)),
                ),
            )

    if crawl_results["accept_button_detected"] == 1:
        if crawl_results["reject_button_detected"] == 0:
            crawl_results[f"accept_button_detected_without_reject_button"] = True
        else:
            crawl_results[f"accept_button_detected_without_reject_button"] = False

        assert isinstance(crawl_results["reject_button_detected"], int)
        if crawl_results["reject_button_detected"] > 1:
            crawl_results[f"obstruction_detected"] = True
        else:
            crawl_results[f"obstruction_detected"] = False

    return crawl_results
