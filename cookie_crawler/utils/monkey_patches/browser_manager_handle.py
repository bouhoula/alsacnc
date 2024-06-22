# Adapted from https://github.com/openwpm/OpenWPM/
import json
import time
from queue import Empty as EmptyQueue

from openwpm.browser_manager import BrowserManagerHandle
from openwpm.commands.browser_commands import FinalizeCommand
from openwpm.commands.utils.webdriver_utils import parse_neterror
from openwpm.storage.storage_providers import TableName
from openwpm.task_manager import TaskManager

from database.queries import insert_exception_into_db

from .command_sequence import CommandSequence


def execute_command_sequence(
    self: BrowserManagerHandle,
    # Quoting to break cyclic import, see https://stackoverflow.com/a/39757388
    task_manager: "TaskManager",
    command_sequence: CommandSequence,
) -> None:
    """
    Sends CommandSequence to the BrowserManager one command at a time
    """
    assert self.browser_id is not None
    assert self.curr_visit_id is not None
    task_manager.sock.store_record(
        TableName("site_visits"),
        self.curr_visit_id,
        {
            "visit_id": self.curr_visit_id,
            "browser_id": self.browser_id,
            "site_url": command_sequence.url,
            "site_rank": command_sequence.site_rank,
        },
    )
    self.is_fresh = False

    reset = command_sequence.reset
    self.logger.info(
        "Starting to work on CommandSequence with " "visit_id %d on browser with id %d",
        self.curr_visit_id,
        self.browser_id,
    )
    assert self.command_queue is not None
    assert self.status_queue is not None

    for command_and_timeout in command_sequence.get_commands_with_timeout():
        command, timeout = command_and_timeout
        command.set_visit_browser_id(self.curr_visit_id, self.browser_id)
        if not hasattr(command, "url"):
            command.url = command_sequence.url
        command.set_start_time(time.time())
        self.current_timeout = timeout

        # Adding timer to track performance of commands
        t1 = time.time_ns()

        # passes off command and waits for a success (or failure signal)
        self.command_queue.put(command)

        # received reply from BrowserManager, either success or failure
        error_text = None
        tb = None
        status = None
        try:
            status = self.status_queue.get(True, self.current_timeout)
        except EmptyQueue:
            self.logger.info(
                "BROWSER %i: Timeout while executing command, %s, killing "
                "browser manager" % (self.browser_id, repr(command))
            )
            insert_exception_into_db(command_sequence.website, "Timeout")

        if status is None:
            # allows us to skip this entire block without having to bloat
            # every if statement
            command_status = "timeout"
            pass
        elif status == "OK":
            command_status = "ok"
        elif status[0] == "CRITICAL":
            command_status = "critical"
            self.logger.critical(
                "BROWSER %i: Received critical error from browser "
                "process while executing command %s. Setting failure "
                "status." % (self.browser_id, str(command))
            )
            task_manager.failure_status = {
                "ErrorType": "CriticalChildException",
                "CommandSequence": command_sequence,
                "Exception": status[1],
            }
            error_text, tb = self._unpack_pickled_error(status[1])
        elif status[0] == "FAILED":
            command_status = "error"
            error_text, tb = self._unpack_pickled_error(status[1])
            self.logger.info(
                "BROWSER %i: Received failure status while executing "
                "command: %s" % (self.browser_id, repr(command))
            )
        elif status[0] == "NETERROR":
            command_status = "neterror"
            error_text, tb = self._unpack_pickled_error(status[1])
            error_text = parse_neterror(error_text)
            self.logger.info(
                "BROWSER %i: Received neterror %s while executing "
                "command: %s" % (self.browser_id, error_text, repr(command))
            )
        else:
            raise ValueError("Unknown browser status message %s" % status)

        task_manager.sock.store_record(
            TableName("crawl_history"),
            self.curr_visit_id,
            {
                "browser_id": self.browser_id,
                "visit_id": self.curr_visit_id,
                "command": type(command).__name__,
                "arguments": json.dumps(
                    command.__dict__, default=lambda x: repr(x)
                ).encode("utf-8"),
                "retry_number": command_sequence.retry_number,
                "command_status": command_status,
                "error": error_text,
                "traceback": tb,
                "duration": int((time.time_ns() - t1) / 1000000),
            },
        )

        if command_status == "critical":
            task_manager.sock.finalize_visit_id(
                success=False,
                visit_id=self.curr_visit_id,
            )
            return

        if command_status != "ok":
            with task_manager.threadlock:
                task_manager.failure_count += 1
            if task_manager.failure_count > task_manager.failure_limit:
                self.logger.critical(
                    "BROWSER %i: Command execution failure pushes failure "
                    "count above the allowable limit. Setting "
                    "failure_status." % self.browser_id
                )
                task_manager.failure_status = {
                    "ErrorType": "ExceedCommandFailureLimit",
                    "CommandSequence": command_sequence,
                }
                return
            self.restart_required = True
            self.logger.debug("BROWSER %i: Browser restart required" % self.browser_id)
        # Reset failure_count at the end of each successful command sequence
        elif type(command) is FinalizeCommand:
            with task_manager.threadlock:
                task_manager.failure_count = 0

        if self.restart_required:
            task_manager.sock.finalize_visit_id(
                success=False, visit_id=self.curr_visit_id
            )
            break

    self.logger.info(
        "Finished working on CommandSequence with " "visit_id %d on browser with id %d",
        self.curr_visit_id,
        self.browser_id,
    )
    # Sleep after executing CommandSequence to provide extra time for
    # internal buffers to drain. Stopgap in support of #135
    time.sleep(2)

    if task_manager.closing:
        return

    if self.restart_required or reset:
        success = self.restart_browser_manager(clear_profile=reset)
        if not success:
            self.logger.critical(
                "BROWSER %i: Exceeded the maximum allowable consecutive "
                "browser launch failures. Setting failure_status." % self.browser_id
            )
            task_manager.failure_status = {
                "ErrorType": "ExceedLaunchFailureLimit",
                "CommandSequence": command_sequence,
            }
            return
        self.restart_required = False


def monkey_patch_browser_manager_handle() -> None:
    BrowserManagerHandle.execute_command_sequence = execute_command_sequence
