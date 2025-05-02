import logging
import sys

# Configure logging to avoid duplicate logs
def setup_logging():
    """
    Set up logging configuration to prevent duplicate logs
    """
    root_logger = logging.getLogger()
    
    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure main logger
    logger = logging.getLogger('digital_transformation')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Important to prevent duplicate logs
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create new console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()
