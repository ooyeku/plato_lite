# utils/logger.py

import logging
import os
from datetime import datetime
from functools import wraps
import time
import inspect

from config import PROJECT_DIR


class ColorFormatter(logging.Formatter):
    """Custom color formatter for console output."""
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'  # Reset color
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"


class PlatoLogger:
    """
    PlatoLogger: A custom logger for the Plato project.

    This logger provides colorized console output and file logging with
    detailed information about the calling context.
    """

    def __init__(self, name='plato_lite', log_file='plato_lite.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(PROJECT_DIR, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log_with_context(self, level, message):
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name

        full_message = f"{filename}:{lineno} - {func_name}() - {message}"
        getattr(self.logger, level)(full_message)

    def debug(self, message):
        self._log_with_context('debug', message)

    def info(self, message):
        self._log_with_context('info', message)

    def warning(self, message):
        self._log_with_context('warning', message)

    def error(self, message):
        self._log_with_context('error', message)

    def critical(self, message):
        self._log_with_context('critical', message)

    @staticmethod
    def log_execution_time(func):
        """Decorator to log function execution time."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
            return result

        return wrapper


# Create a global logger instance
logger = PlatoLogger()

# Example usage
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


    @logger.log_execution_time
    def slow_function():
        time.sleep(2)
        logger.info("Slow function executed")


    slow_function()