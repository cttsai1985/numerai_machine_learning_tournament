import logging
import sys

formatted_string = "%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(formatted_string)
basic_configs = {
    "level": logging.DEBUG,
    "format": formatted_string,
    "datefmt": '%Y-%m-%d %H:%M:%S',
}


def initialize_logger():
    logging.basicConfig(level=logging.DEBUG, format=formatted_string, datefmt='%Y-%m-%d %H:%M:%S', )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def add_file_logger(log_filename: str):
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    return logger


if "__main__" == __name__:
    initialize_logger()
