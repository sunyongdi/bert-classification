import time
import logging

logger = logging.getLogger(__name__)

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        # print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper