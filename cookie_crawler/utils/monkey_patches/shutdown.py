# Adapted from https://github.com/openwpm/OpenWPM/
import asyncio

from openwpm.storage.storage_controller import StorageController


async def shutdown(
    self: StorageController, completion_queue_task: asyncio.Task[None]
) -> None:
    completion_tokens = {}
    visit_ids = list(self.store_record_tasks.keys())
    for visit_id in visit_ids:
        t = await self.finalize_visit_id(visit_id, success=False)
        if t is not None:
            completion_tokens[visit_id] = t
    await self.structured_storage.flush_cache()

    try:
        await asyncio.wait_for(completion_queue_task, timeout=1800)
    except asyncio.TimeoutError:
        self.logger.info("Completion queue task timed out")

    try:
        for visit_id, token in completion_tokens.items():
            await asyncio.wait_for(token, timeout=1800)
            self.completion_queue.put((visit_id, False))
    except asyncio.TimeoutError:
        self.logger.info("Completion tokens timed out")

    try:
        await self.structured_storage.shutdown()
    except asyncio.TimeoutError:
        self.logger.info("Time out in StorageProvider shutdown")

    if self.unstructured_storage is not None:
        await self.unstructured_storage.flush_cache()
        await self.unstructured_storage.shutdown()
