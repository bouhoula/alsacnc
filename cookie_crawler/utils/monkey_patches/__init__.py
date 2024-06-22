import openwpm

from .browser_manager import monkey_patch_browser_manager
from .browser_manager_handle import monkey_patch_browser_manager_handle
from .shutdown import shutdown


def apply_monkey_patches() -> None:
    monkey_patch_browser_manager()
    monkey_patch_browser_manager_handle()


def apply_monkey_patch_to_task_manager(
    manager: openwpm.task_manager.TaskManager,
) -> None:
    manager.storage_controller_handle.storage_controller.shutdown = shutdown


def set_screen_dimensions(width: int, height: int) -> None:
    """This is a temporary solution until OpenWPM allows for custom resolutions
    using BrowserParams
    """
    openwpm.deploy_browsers.deploy_firefox.DEFAULT_SCREEN_RES = (width, height)
