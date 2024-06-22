import logging
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple
from urllib import parse as urlparse

from openwpm.commands.browser_commands import close_other_windows, tab_restart_browser
from openwpm.commands.types import BaseCommand
from openwpm.commands.utils.webdriver_utils import (
    is_displayed,
    scroll_down,
    wait_until_loaded,
)
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket

from selenium.common.exceptions import (
    MoveTargetOutOfBoundsException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver import Firefox
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from cookie_crawler.utils.domains import get_prefix_list
from cookie_crawler.utils.js import (
    clear_data,
    click,
    contains,
    find_element_by_selector,
    get_selector_from_element,
    scroll_to_bottom,
)
from shared_utils import repeat, url_to_uniform_domain

# Constants for bot mitigation
NUM_MOUSE_MOVES = 10  # Times to randomly move the mouse
RANDOM_SLEEP_LOW = 1  # low (in sec) for random sleep between page loads
RANDOM_SLEEP_HIGH = 7  # high (in sec) for random sleep between page loads

logger = logging.getLogger("openwpm")


def bot_mitigation(
    webdriver: Firefox,
    random_moving: bool,
    random_scrolling: bool,
    random_waiting: bool,
) -> None:
    """This function only differs from the one provided in OpenWPM by disabling scrolling,
    as this leads to screenshots being displaced."""

    # bot mitigation 1: move the randomly around a number of times
    if random_moving:
        window_size = webdriver.get_window_size()
        num_moves = 0
        num_fails = 0
        while num_moves < NUM_MOUSE_MOVES + 1 and num_fails < NUM_MOUSE_MOVES:
            try:
                if num_moves == 0:  # move to the center of the screen
                    x = int(round(window_size["height"] / 2))
                    y = int(round(window_size["width"] / 2))
                else:  # move a random amount in some direction
                    move_max = random.randint(0, 500)
                    x = random.randint(-move_max, move_max)
                    y = random.randint(-move_max, move_max)
                action = ActionChains(webdriver)
                action.move_by_offset(x, y)
                action.perform()
                num_moves += 1
            except MoveTargetOutOfBoundsException:
                num_fails += 1
                pass

    # bot mitigation 2: scroll in random intervals down page
    if random_scrolling:
        scroll_down(webdriver)

    # bot mitigation 4: randomly wait so page visits happen with irregularity
    if random_waiting:
        time.sleep(random.randrange(RANDOM_SLEEP_LOW, RANDOM_SLEEP_HIGH))


def get_command(
    url: str,
    sleep: float,
    webdriver: Firefox,
    browser_params: BrowserParams,
    close_dialog_if_exists: bool = True,
    random_moving: bool = True,
    random_scrolling: bool = False,
    random_waiting: bool = False,
) -> None:
    tab_restart_browser(webdriver)

    try:
        webdriver.get(url)
    except TimeoutException:
        pass

    time.sleep(sleep)

    if close_dialog_if_exists:
        try:
            WebDriverWait(webdriver, 0.5).until(EC.alert_is_present())
            alert = webdriver.switch_to.alert
            alert.dismiss()
            time.sleep(1)
        except (TimeoutException, WebDriverException):
            pass

    close_other_windows(webdriver)

    if browser_params.bot_mitigation:
        bot_mitigation(webdriver, random_moving, random_scrolling, random_waiting)


def is_stale(element: WebElement) -> bool:
    try:
        _ = element.text
        return False
    except StaleElementReferenceException:
        return True


@repeat()
def reload_page_and_click_on_elements(
    webdriver: Firefox,
    elements_ids: List[str],
    iframes_ids: List[str],
    url: str,
    browser_params: BrowserParams,
    delete_cookies: bool = False,
    sleep: int = 5,
) -> Tuple[Tuple[datetime, Optional[datetime]], bool]:
    if delete_cookies:
        clear_data(webdriver)
        time.sleep(4)
    reload_timestamp = datetime.utcnow()
    click_timestamp = None
    time.sleep(1)
    get_command(url, sleep, webdriver, browser_params)
    body = webdriver.find_element("tag name", "body")
    for element_id, iframe_id in zip(elements_ids, iframes_ids):
        if iframe_id is not None:
            iframe = find_element_by_selector(iframe_id, webdriver)
            webdriver.switch_to.frame(iframe)

        element = find_element_by_selector(element_id, webdriver)
        if element is None:
            raise ValueError(
                "Could not find element to click. It is likely that the webpage was not loaded properly."
            )
        time.sleep(1)
        click_timestamp = datetime.utcnow()
        click(element, webdriver)
        if iframe_id is not None:
            webdriver.switch_to.parent_frame()
    time.sleep(2)
    return (reload_timestamp, click_timestamp), is_stale(body) or len(
        webdriver.window_handles
    ) > 1


def load_page(
    url: str,
    webdriver: Firefox,
    browser_params: BrowserParams,
    delete_cookies: bool = False,
    sleep: int = 5,
) -> Tuple[datetime, bool]:
    (load_timestamp, _), makes_banner_disappear = reload_page_and_click_on_elements(
        webdriver, [], [], url, browser_params, delete_cookies, sleep
    )
    return load_timestamp, makes_banner_disappear


def find_prefix_and_load_page(
    url: str,
    webdriver: Firefox,
    browser_params: BrowserParams,
    sleep: int = 5,
) -> Tuple[Optional[str], Optional[datetime]]:
    prefix_list = get_prefix_list(url)
    for i, prefix in enumerate(prefix_list):
        try:
            load_timestamp, _ = load_page(
                f"{prefix}{url}", webdriver, browser_params, sleep=sleep
            )
            return prefix, load_timestamp
        except:
            pass
    return None, None


def get_intra_links(webdriver: Firefox, url: str) -> List[WebElement]:
    ps1 = url_to_uniform_domain(url)
    links = list()
    for elem in webdriver.find_elements(By.TAG_NAME, "a"):
        try:
            href = elem.get_attribute("href")
        except StaleElementReferenceException:
            continue
        if href is None:
            continue
        full_href = urlparse.urljoin(url, href)
        if not full_href.startswith("http"):
            continue
        if url_to_uniform_domain(full_href) == ps1:
            links.append(elem)
    return links


def browse(
    website: str,
    webdriver: Firefox,
    num_links: int,
    sleep: int,
    seed: int,
    excluded_element_id: Optional[str] = None,
) -> None:
    excluded_element = (
        find_element_by_selector(excluded_element_id, webdriver)
        if excluded_element_id is not None
        else None
    )
    links = [
        x
        for x in get_intra_links(webdriver, website)
        if is_displayed(x) is True
        and (excluded_element is None or not contains(excluded_element, x, webdriver))
    ]
    if not links:
        return
    links_ids = get_selector_from_element(links, webdriver)
    random.seed(seed)
    for i in range(num_links):
        r = int(random.random() * len(links))

        try:
            link = (
                find_element_by_selector(links_ids[r], webdriver) if i > 0 else links[r]
            )
            if link is None:
                continue
            click(link, webdriver)
            wait_until_loaded(webdriver, 4)
            time.sleep(0.5)
            scroll_to_bottom(webdriver)
            wait_until_loaded(webdriver, 3)
            time.sleep(0.5)
            webdriver.back()
        except Exception:
            pass


class GetCommand(BaseCommand):
    """
    goes to <url> using the given <webdriver> instance
    """

    def __init__(self, url: str, sleep: float):
        self.url = url
        self.sleep = sleep

    def __repr__(self) -> str:
        return "GetCommand({},{})".format(self.url, self.sleep)

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        get_command(
            self.url,
            self.sleep,
            webdriver,
            browser_params,
        )
