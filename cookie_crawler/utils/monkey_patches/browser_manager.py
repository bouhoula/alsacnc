# Adapted from https://github.com/openwpm/OpenWPM/
import pickle
import sys
import time
import traceback
from typing import Dict, Optional, Union

from openwpm.browser_manager import BrowserManager
from openwpm.commands.types import BaseCommand, ShutdownSignal
from openwpm.deploy_browsers import deploy_firefox
from openwpm.socket_interface import ClientSocket
from openwpm.utilities.multiprocess_utils import parse_traceback_for_sentry

from selenium.common.exceptions import WebDriverException

from cookie_crawler.commands.cookie_banner_command import GetCookieBannerCommand
from database.queries import insert_exception_into_db


def run(self: BrowserManager) -> None:
    assert self.browser_params.browser_id is not None
    display = None
    cur_website: Optional[Dict] = None

    try:
        # Start Xvfb (if necessary), webdriver, and browser
        driver, browser_profile_path, display = deploy_firefox.deploy_firefox(
            self.status_queue,
            self.browser_params,
            self.manager_params,
            self.crash_recovery,
        )

        extension_socket: Optional[ClientSocket] = None

        if self.browser_params.extension_enabled:
            extension_socket = self._start_extension(browser_profile_path)

        self.logger.debug(
            "BROWSER %i: BrowserManager ready." % self.browser_params.browser_id
        )

        # passes "READY" to the TaskManager to signal a successful startup
        self.status_queue.put(("STATUS", "Browser Ready", "READY"))
        self.browser_params.profile_path = browser_profile_path

        assert extension_socket is not None
        # starts accepting arguments until told to die
        while True:
            # no command for now -> sleep to avoid pegging CPU on blocking get
            if self.command_queue.empty():
                time.sleep(0.001)
                continue

            command: Union[ShutdownSignal, BaseCommand] = self.command_queue.get()

            if isinstance(command, GetCookieBannerCommand):
                cur_website = command.website

            if isinstance(command, ShutdownSignal):
                driver.quit()
                self.status_queue.put("OK")
                return

            assert isinstance(command, BaseCommand)
            self.logger.info(
                "BROWSER %i: EXECUTING COMMAND: %s"
                % (self.browser_params.browser_id, str(command))
            )

            # attempts to perform an action and return an OK signal
            # if command fails for whatever reason, tell the TaskManager to
            # kill and restart its worker processes
            try:
                command.execute(
                    driver,
                    self.browser_params,
                    self.manager_params,
                    extension_socket,
                )
                self.status_queue.put("OK")
            except WebDriverException:
                # We handle WebDriverExceptions separately here because they
                # are quite common, and we often still have a handle to the
                # browser, allowing us to run the SHUTDOWN command.
                tb = traceback.format_exception(*sys.exc_info())
                if "about:neterror" in tb[-1]:
                    self.status_queue.put(("NETERROR", pickle.dumps(sys.exc_info())))
                    continue
                extra = parse_traceback_for_sentry(tb)
                extra["exception"] = tb[-1]
                self.logger.error(
                    "BROWSER %i: WebDriverException while executing command"
                    % self.browser_params.browser_id,
                    exc_info=True,
                    extra=extra,
                )
                self.status_queue.put(("FAILED", pickle.dumps(sys.exc_info())))
                if cur_website is not None:
                    insert_exception_into_db(cur_website, "Webdriver exception thrown")

    except self.critical_exceptions as e:
        self.logger.error(
            "BROWSER %i: %s thrown, informing parent and raising"
            % (self.browser_params.browser_id, e.__class__.__name__)
        )
        self.status_queue.put(("CRITICAL", pickle.dumps(sys.exc_info())))
        if cur_website is not None:
            insert_exception_into_db(cur_website, "Critical exception thrown")
    except Exception:
        tb = traceback.format_exception(*sys.exc_info())
        extra = parse_traceback_for_sentry(tb)
        extra["exception"] = tb[-1]
        self.logger.error(
            "BROWSER %i: Crash in driver, restarting browser manager"
            % self.browser_params.browser_id,
            exc_info=True,
            extra=extra,
        )
        self.status_queue.put(("FAILED", pickle.dumps(sys.exc_info())))
        if cur_website is not None:
            insert_exception_into_db(cur_website, "Crash in driver")
    finally:
        if display is not None:
            display.stop()
        return


def monkey_patch_browser_manager() -> None:
    BrowserManager.run = run
