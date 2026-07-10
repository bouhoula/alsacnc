import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

from openwpm.commands.types import BaseCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket

import langdetect
from selenium.webdriver import Firefox

import cookie_crawler.commands.explore_cookie_banner as explore
from cookie_crawler.commands.detect_cookie_banner import detect_cookie_banner
from cookie_crawler.commands.detect_dark_patterns import detect_forced_action
from cookie_crawler.commands.get_command import (
    browse,
    find_prefix_and_load_page,
    load_page,
    scroll_to_bottom,
)
from cookie_crawler.utils.cmp import detect_cmp
from cookie_crawler.utils.css_selectors import parse_selectors_for_url
from cookie_crawler.utils.gpc import detect_gpc
from cookie_crawler.utils.js import (
    extract_text_from_element,
    find_element_by_selector,
    get_link_to_text_ratio,
    get_selector_from_element,
    get_z_index,
)
from cookie_crawler.utils.report_screenshots import (
    highlight_and_capture,
    save_report_screenshot,
)
from cookie_crawler.utils.translate import detect_language
from database.queries import insert_exception_into_db, insert_into_db, update_entry
from shared_utils import get_timestamp, write_to_file

logger = logging.getLogger("openwpm")


class GetCookieBannerCommand(BaseCommand):
    def __init__(self, website: Dict, config: Dict):
        self.website = website
        self.config = config

    def __repr__(self) -> str:
        return (
            f"GetCookieBannerCommand({self.website['url']}, "
            f"{self.website['crux_country']}, rank={self.website['crux_rank']})"
        )

    def _capture_before_exploration_screenshot(
        self,
        webdriver: Firefox,
        banner_selector: Optional[str],
        iframe_id: Optional[str],
    ) -> None:
        switched = False
        try:
            if iframe_id is not None:
                iframe_el = find_element_by_selector(iframe_id, webdriver)
                if iframe_el is not None:
                    webdriver.switch_to.frame(iframe_el)
                    switched = True
            banner_el = (
                find_element_by_selector(banner_selector, webdriver)
                if banner_selector is not None
                else None
            )
            if banner_el is not None:
                highlight_and_capture(
                    webdriver,
                    banner_el,
                    self.website["save_path"],
                    "No interaction",
                    0,
                    highlighted_text=None,
                )
            else:
                save_report_screenshot(
                    webdriver, self.website["save_path"], "No interaction", 0
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not capture avant_toute_action screenshot: {e}")
        finally:
            if switched:
                webdriver.switch_to.parent_frame()

    # flake8: noqa: C901
    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        if extension_socket is not None:
            extension_socket.send(self.visit_id)

        prefix, load_timestamp = find_prefix_and_load_page(
            self.website["url"], webdriver, browser_params
        )
        print(get_timestamp(self.website["name"]), "prefix", prefix)

        if prefix is None:
            insert_exception_into_db(self.website, "Unable to load page.")
            return

        self.website["url"] = f"{prefix}{self.website['url']}"

        dfs_depth = 0
        if self.config["explore_cookie_banner"]:
            dfs_depth = self.config["dfs_depth"]
            if self.website["previously_timed_out"]:
                if dfs_depth == 1:
                    insert_exception_into_db(self.website, "Timeout")
                    return
                dfs_depth = dfs_depth - 1
                logger.info(
                    f"Crawling {self.website['name']} has previously timed out. "
                    f"Setting exploration depth to {dfs_depth}."
                )

        body = webdriver.find_element("tag name", "body")
        try:
            self.website["language"] = langdetect.detect(
                extract_text_from_element(body, webdriver)
            )
        except:
            pass

        if self.website.get("language", None) not in self.config["supported_languages"]:
            self.website["other"] = dict(language_not_supported=True)
            update_entry(self.website)
            print(
                get_timestamp(self.website["name"]),
                "Unsupported language detected:",
                self.website["language"],
            )
            return

        if self.config["check_gpc_presence"]:
            gpc_detected = detect_gpc(self.website["name"])
            logger.info(f"GPC detected: {gpc_detected}")
            self.website["gpc_detected"] = int(gpc_detected)

        detect_cmp(self.website, webdriver, **self.config["detect_cmp"])
        self.website["visit_id"] = self.visit_id
        update_entry(self.website)

        selectors = parse_selectors_for_url(
            self.website["name"], *browser_params.custom_params["selectors"]
        )
        cookie_banner = insert_into_db(
            "cookie_banners",
            dict(
                website_id=self.website["id"],
                dfs_max_depth=dfs_depth,
            ),
        )

        start_time = time.time()

        (
            elements,
            detected_selectors,
            iframes,
            z_index_based_detection,
        ) = detect_cookie_banner(
            webdriver,
            self.website["language"],
            self.config["supported_languages"],
            use_z_index=self.config["extract_cookie_settings_with_z_index"],
            explore_iframes=True,
            candidate_selectors=selectors,
            depth=0,
        )

        execution_times = dict(
            website_id=self.website["id"], notice_detection=time.time() - start_time
        )

        iframe_id = None
        if len(elements) > 0:
            element = elements[0]
            iframe = iframes[0]

            iframe_id = (
                None if iframe is None else get_selector_from_element(iframe, webdriver)
            )
            if iframe is not None:
                webdriver.switch_to.frame(iframe)

            cookie_banner_html = element.get_attribute("innerHTML")
            cookie_banner_text = extract_text_from_element(element, webdriver)

            cookie_banner.update(
                width=int(element.size["width"]),
                height=int(element.size["height"]),
                position_x=int(element.location["x"]),
                position_y=int(element.location["y"]),
                selector="\n".join(detected_selectors),
                link_to_text_ratio=get_link_to_text_ratio(element, webdriver),
                text=cookie_banner_text,
            )

            text_language = detect_language(cookie_banner_text)
            if text_language in self.config["supported_languages"]:
                self.website["language"] = text_language
                update_entry(self.website)

            cookie_banner.update(
                html=cookie_banner_html,
                detected=1,
                detection_method=("z-index" if z_index_based_detection else "selector"),
                cookie_mention=1,
                z_index=get_z_index(element, webdriver, default_value=None),
            )
            #print ("Save Banner : " + str(self.config["save_cookie_banner_screenshot"]))
            if self.config["save_cookie_banner_screenshot"]:
                element.screenshot(
                    os.path.join(
                        self.website["save_path"], "cookie_banner_screenshot.png"
                    )
                )
                #self.website["screenshot"] = webdriver.get_full_page_screenshot_as_png()
                self.website["screenshot"] = webdriver.get_screenshot_as_png()
                update_entry(self.website)
                webdriver.save_screenshot(
                    os.path.join(
                        self.website["save_path"], "full_screenshot.png"
                    )
                )

            if iframe_id is not None:
                webdriver.switch_to.parent_frame()

        if not cookie_banner["detected"]:
            cookie_banner["cookie_mention"] = int(
                "cookie" in webdriver.find_element("xpath", "//html/body").text.lower()
            )

        update_entry(cookie_banner)
        logger.info(get_timestamp(self.website["name"]))
        logger.info(
            f"Language: {self.website['language']}\n"
            f"Cookie notice detected: {bool(cookie_banner['detected'])}\n"
            f"Text: {repr(cookie_banner['text'])}\n"
        )

        if self.config["extract_accept_none_cookies"]:
            try:
                start_time = time.time()
                cookies = webdriver.get_cookies()
                insert_into_db(
                    "cookie_timestamps",
                    dict(
                        website_id=self.website["id"],
                        collection_strategy="Before Exploration",
                        visit_id=self.visit_id,
                        start_timestamp=load_timestamp,
                        end_timestamp=datetime.utcnow(),
                        num_cookies=len(cookies),
                    ),
                )
                banner_selector = (
                    detected_selectors[0] if cookie_banner["detected"] else None
                )
                print("Scrolling ..")
                scroll_to_bottom(webdriver)
                time.sleep(2)
                cookies = webdriver.get_cookies()
                insert_into_db(
                    "cookie_timestamps",
                    dict(
                        website_id=self.website["id"],
                        collection_strategy="After scrolling",
                        visit_id=self.visit_id,
                        start_timestamp=load_timestamp,
                        end_timestamp=datetime.utcnow(),
                        num_cookies=len(cookies),
                    ),
                )
                print("Will start browsing")
                browse(
                    self.website["url"],
                    webdriver,
                    num_links=self.config["num_links_for_browsing"],
                    sleep=1,
                    seed=42,
                    excluded_element_id=banner_selector if iframe_id is None else None,
                )
                cookies = webdriver.get_cookies()
                insert_into_db(
                    "cookie_timestamps",
                    dict(
                        website_id=self.website["id"],
                        collection_strategy="No interaction",
                        visit_id=self.visit_id,
                        start_timestamp=load_timestamp,
                        end_timestamp=datetime.utcnow(),
                        num_cookies=len(cookies),
                    ),
                )
                if self.config.get("save_report_screenshots"):
                    self._capture_before_exploration_screenshot(
                        webdriver, banner_selector, iframe_id
                    )
                load_timestamp, _ = load_page(
                    self.website["url"], webdriver, browser_params, delete_cookies=True
                )
                execution_times["initial_cookie_extraction"] = time.time() - start_time
            except Exception as e:
                logger.error(f"Error caught during cookie extraction: {e}")
                insert_exception_into_db(self.website, e)

        if self.config["dump_full_page_html"]:
            page_source = webdriver.page_source.encode("utf8")
            write_to_file(
                page_source,
                os.path.join(self.website["save_path"], "cookie_banner.html"),
            )

        forced_action_detected: Optional[bool] = None
        if cookie_banner["detected"] and self.config["detect_dark_patterns"]:
            forced_action_detected = detect_forced_action(
                detected_selectors[0], iframe_id, webdriver
            )
        crawl_results = insert_into_db(
            "crawl_results",
            dict(
                website_id=self.website["id"],
                cookie_notice_detected=bool(cookie_banner["detected"]),
                forced_action_detected=forced_action_detected,
            ),
        )

        if cookie_banner["detected"] and self.config["explore_cookie_banner"]:
            exploration_modes = self.config["exploration_modes"]
            assert all(
                mode in ["naive", "with_ietc_model"] for mode in exploration_modes
            )
            for mode in exploration_modes:
                try:
                    start_time = time.time()
                    load_timestamp, _ = load_page(
                        self.website["url"],
                        webdriver,
                        browser_params,
                        delete_cookies=True,
                    )
                    additional_args = (
                        dict(max_depth=dfs_depth, load_timestamp=load_timestamp)
                        if mode == "naive"
                        else {}
                    )
                    crawl_results.update(
                        **getattr(explore, f"explore_cookie_banner_{mode}")(
                            detected_selectors[0],
                            iframe_id,
                            self.website,
                            webdriver,
                            browser_params,
                            self.config,
                            **additional_args,
                        )
                    )
                    execution_times[f"exploration_{mode}"] = time.time() - start_time
                except Exception as e:
                    logger.error(
                        f"Error caught during cookie banner exploration `{mode}`: {e}"
                    )
                    insert_exception_into_db(self.website, e)

                update_entry(crawl_results)

        insert_into_db("execution_times", execution_times)
