import structlog
import logging
import sys
import os

def setup_logger(log_file=None, terminal_level=logging.ERROR, file_level=logging.WARNING):
    """Configures structlog with colored terminal output and plain text file output."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        
    handlers = []
    
    # Shared processors for both formatters
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%d-%m-%Y %H:%M:%S"),
    ]
    
    # Terminal Handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(terminal_level)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File Handler without colors (plain text)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=shared_processors,
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
    logging.basicConfig(level=min(terminal_level, file_level), handlers=handlers)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%d-%m-%Y %H:%M:%S"),
            # REMOVED format_exc_info and StackInfoRenderer to fix the warning
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
