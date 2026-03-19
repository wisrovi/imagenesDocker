"""DVC file processing worker module.

This module provides a worker function that processes files through the DVC
pipeline including cleaning, verification, adding, pushing, and inventory creation.
"""

import os
import subprocess
from typing import Dict

from loguru import logger

from src.scripts.api.app.worker_queue import (
    get_from_queue,
    start_receiving,
    update_queue,
)

# Processing flags - toggle steps on/off
CLEAN: bool = True
VERIFY: bool = False
ADD: bool = True
PUSH: bool = True
INVENTORY: bool = True

# DVC command definitions
CLEAN_COMMAND = "rm -rf /app/.dvc/tmp && rm -rf /app/.dvc/cache"
REPRO_COMMAND = "dvc repro "
VERIFY_COMMAND = "dvc status "
ADD_COMMAND = "dvc add "
PUSH_COMMAND = "dvc push"
AFTER_PUSH_COMMAND = "git rm -r --cached "
AFTER_PUSH_COMMAND_2 = 'git commit -m "stop tracking" '
INVENTORY_COMMAND = "python /app/src/scripts/upload/files_inventory.py "


def worker_function(path: str, metadata: Dict) -> bool:
    """Processes each item from the DVC queue.

    Executes the complete DVC pipeline: clean, verify, add, push, and inventory.

    Args:
        path: The file system path to process.
        metadata: Dictionary containing task metadata and status information.

    Returns:
        True if processing completed successfully, False otherwise.
    """
    task_id = metadata.get("id")
    if not task_id:
        logger.error("No task ID found in metadata")
        return False

    if path.endswith(os.sep):
        path = path[:-1]

    push_path = f"'{path}.dvc'"
    path = f"'{path}'"

    print("\n" * 5)
    print("*" * 30)
    print(f"New item received for push with dvc: {path}")
    print("*" * 30)
    print("\n")

    completed = False
    if CLEAN:
        try:
            logger.info(f"Cleaning DVC temporary files before processing {path}.")
            subprocess.run(CLEAN_COMMAND, shell=True, check=True, capture_output=True)
            logger.info("Temporary files cleaned.")

            metadata["metadata"]["detail"] = "1/7 -> Cleaned"
        except subprocess.CalledProcessError:
            print(f"Data at {path} is not tracked by DVC. Adding and pushing...")
        update_queue(str(task_id), metadata)

        try:
            logger.info(f"Reproducing DVC pipeline before processing {path}.")
            subprocess.run(REPRO_COMMAND, shell=True, check=True, capture_output=True)
            logger.info("Pipeline reproduced.")

            metadata["metadata"]["detail"] = "1/7 -> Reproduced"
        except subprocess.CalledProcessError:
            print(f"Data at {path} is not tracked by DVC. Adding and pushing...")
        update_queue(str(task_id), metadata)

    if not completed:
        if VERIFY:
            try:
                logger.info(f"Starting verification before adding {path} to DVC.")
                executor_command = f"{VERIFY_COMMAND} {path}"
                print(executor_command)
                subprocess.run(
                    executor_command, shell=True, check=True, capture_output=True
                )
                logger.info("Verification completed")

                metadata["metadata"]["detail"] = "2/7 -> Verified existence"
            except subprocess.CalledProcessError as e:
                print(f"Error during verification: {e.stderr.decode().strip()}")
                metadata["metadata"]["status"] = "failed"
                metadata["metadata"]["detail"] = "2/7 -> " + str(e)
                update_queue(str(task_id), metadata)
                return False

            update_queue(str(task_id), metadata)

        if ADD:
            try:
                logger.info(f"Adding {path} to DVC.")
                executor_command = f"{ADD_COMMAND} {path}"
                subprocess.run(
                    executor_command, shell=True, check=True, capture_output=True
                )
                logger.info("DVC add completed")

                metadata["metadata"]["detail"] = "3/7 -> DVC add"
            except subprocess.CalledProcessError as e:
                print(f"Error during DVC add: {e.stderr.decode().strip()}")
                metadata["metadata"]["status"] = "failed"
                metadata["metadata"]["detail"] = "3/7 -> " + str(e)
                update_queue(str(task_id), metadata)
                return False
            update_queue(str(task_id), metadata)

        if PUSH:
            try:
                logger.info(f"Pushing {path} to remote DVC storage.")
                executor_command = f"{PUSH_COMMAND} {push_path}"
                subprocess.run(
                    executor_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                completed = True
                logger.info("DVC push completed")

                metadata["metadata"]["detail"] = "4/7 -> DVC push"
            except subprocess.CalledProcessError as e:
                print(f"Error during DVC push: {e.stderr.decode().strip()}")
                metadata["metadata"]["status"] = "failed"
                metadata["metadata"]["detail"] = "4/7 ->" + str(e)
                update_queue(str(task_id), metadata)
                return False
            update_queue(str(task_id), metadata)

            try:
                logger.info(f"Git removing tracking for {path}")
                executor_command = f"{AFTER_PUSH_COMMAND} {path}"
                subprocess.run(
                    executor_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                logger.info("Git remove completed")

                metadata["metadata"]["detail"] = "5/7 -> Git remove tracking"
            except subprocess.CalledProcessError as e:
                print(f"Error during Git remove: {e.stderr.decode().strip()}")
            update_queue(str(task_id), metadata)

            try:
                logger.info(f"Git committing changes for {path}")
                executor_command = f"{AFTER_PUSH_COMMAND_2} {path}"
                subprocess.run(
                    executor_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                logger.info("Git commit completed")

                metadata["metadata"]["detail"] = "6/7 -> Git commit"
            except subprocess.CalledProcessError as e:
                print(f"Error during Git commit: {e.stderr.decode().strip()}")
            update_queue(str(task_id), metadata)

        if INVENTORY and completed:
            try:
                logger.info(f"Creating inventory for {path} of uploaded dataset")
                executor_command = f"{INVENTORY_COMMAND} {path}"
                print(executor_command)
                subprocess.run(
                    executor_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                logger.info("Inventory creation completed")

                metadata["metadata"]["detail"] = "7/7 -> Created inventory (CSV)"
            except subprocess.CalledProcessError as e:
                print(f"Error creating inventory: {e.stderr.decode().strip()}")
                metadata["metadata"]["status"] = "failed"
                metadata["metadata"]["detail"] = "7/7 -> " + str(e)
                update_queue(str(task_id), metadata)
                return False
            update_queue(str(task_id), metadata)

    logger.info(f"Process completed for {path}")
    return completed


if __name__ == "__main__":
    print("Worker started and waiting for tasks...")

    get_from_queue(worker_function)
    start_receiving()
