#!/usr/bin/env python3
"""
Logging utilities cho LegalAdvisor
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "legaladvisor", log_file: str = None, level: int = logging.INFO):
    """
    Thiết lập logger cho ứng dụng

    Args:
        name: Tên logger
        log_file: Đường dẫn file log (optional)
        level: Mức độ log

    Returns:
        Logger object
    """

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler cho console
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout
        for handler in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Handler cho file (nếu có)
    if log_file:
        # Tạo thư mục log nếu chưa có
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and Path(getattr(handler, 'baseFilename', '')) == log_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def get_logger(name: str = "legaladvisor"):
    """Lấy logger đã được thiết lập"""
    return logging.getLogger(name)

# Thiết lập logger mặc định
default_log_file = f"logs/legaladvisor_{datetime.now().strftime('%Y%m%d')}.log"
logger = setup_logger(log_file=default_log_file)

def log_function_call(func_name: str, args: dict = None, start_time: float = None):
    """Log function call với performance metrics"""

    if args:
        logger.info(f"Calling {func_name} with args: {args}")
    else:
        logger.info(f"Calling {func_name}")

    if start_time is not None:
        import time
        duration = time.time() - start_time
        logger.info(f"{func_name} completed in {duration:.2f}s")

def log_error(error_msg: str, exc_info: bool = True):
    """Log error message"""
    logger.error(error_msg, exc_info=exc_info)

def log_performance(operation: str, duration: float, metadata: dict = None):
    """Log performance metrics"""

    msg = f"{operation} took {duration:.2f}s"
    if metadata:
        msg += f" - Metadata: {metadata}"

    logger.info(msg)

# Decorators for automatic logging
def log_execution(func):
    """Decorator để log execution của function"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting execution of {func_name}")

        import time
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func_name} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func_name} failed after {duration:.2f}s", exc_info=True)
            raise e

    return wrapper

if __name__ == "__main__":
    # Test logger
    logger.info("Testing LegalAdvisor logger")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test performance logging
    import time
    start = time.time()
    time.sleep(0.1)
    log_performance("test_operation", time.time() - start, {"param": "value"})
