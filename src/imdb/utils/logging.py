import structlog
import logging
import sys

def setup_logger(log_file=None, terminal_level=logging.ERROR, file_level=logging.WARNING):
    """Configures structlog for the entire application, with target outputs."""
    # Reset existing handlers so we can reconfigure per-model
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        
    handlers = []
    
    # Terminal Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(terminal_level)
    handlers.append(console_handler)
    
    # File Handler
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        handlers.append(file_handler)
        
    logging.basicConfig(level=min(terminal_level, file_level), handlers=handlers)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)  # Clean, colored output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
