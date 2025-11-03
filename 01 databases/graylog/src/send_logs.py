import logging
from pygelf import GelfUdpHandler

# Configure the logger
logger = logging.getLogger("graylog_logger")
logger.setLevel(logging.DEBUG)

# Add the GELF handler
# Make sure the IP and port match the UDP input in your Graylog instance
handler = GelfUdpHandler(
    host="127.0.0.1",
    port=12201,
    facility="Hola mundo",
    extra_fields={
        "microservice": "queso",
        "version": "pan",
    },
)
logger.addHandler(handler)


print("Sending test logs to Graylog...")

# Send some log messages with integrated tag
logger.debug("This is a debug message 4.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")

print("Logs sent. Check your Graylog instance.")
