import os
import time
import sys
from loguru import logger
from pygelf import GelfUdpHandler

# Configuration
GRAYLOG_HOST = os.getenv("GRAYLOG_HOST", "127.0.0.1")
GRAYLOG_PORT = int(os.getenv("GRAYLOG_PORT", "12201"))

handler = GelfUdpHandler(
    host=GRAYLOG_HOST,
    port=GRAYLOG_PORT,
    facility="Hola mundo",
    extra_fields={
        "microservice": "queso",
        "version": "pan",
    },
    level="DEBUG",
)

# Configure loguru
logger.remove()
logger.add(sys.stdout, format="{time} - {level} - {message}", level="INFO")
logger.add(handler)

# Send logs with tag
logger.info("Starting log sender with loguru. Press Ctrl+C to stop.")
try:
    with logger.contextualize(tag="integrated_tag"):
        logger.debug("Sample debug log.")
        logger.info("Sample info log.")
        logger.warning("Sample warning log.")
        logger.error("Sample error log.")
        logger.critical("Sample critical log.")
except KeyboardInterrupt:
    logger.info("Log sender stopped.")
