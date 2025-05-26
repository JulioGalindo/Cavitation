#!/usr/bin/env python3
"""
demo_logging_complete.py

Full end-to-end test of the Logger utility:
 1. Configures a logger to write both to console and to a file.
 2. Emits messages at all standard levels.
 3. Catches and logs an exception with traceback.
 4. Reads back the log file to verify its contents.
"""

import os
import logging
from utils.logging import Logger

def main():
    # 1) Prepare log directory and file path
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "demo_complete.log")

    # 2) Create a logger instance
    log = Logger.get(
        name="demo_complete",
        level=logging.DEBUG,
        log_file=log_file
    )

    # 3) Emit one message at each level
    log.debug("DEBUG level message")
    log.info("INFO level message")
    log.warning("WARNING level message")
    log.error("ERROR level message")
    log.critical("CRITICAL level message")

    # 4) Simulate and log an exception
    try:
        1 / 0
    except ZeroDivisionError as exc:
        log.exception("Exception caught during computation")

    # 5) Read back and print the log file contents
    print("\n=== Contents of demo_complete.log ===")
    with open(log_file, "r") as f:
        for line in f:
            print(line.rstrip())

if __name__ == "__main__":
    main()
