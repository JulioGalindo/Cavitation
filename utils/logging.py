"""
utils/logging.py

Logging helper for cavitation detection project.
"""
import logging
import os

class Logger:
    """
    Wrapper around Python's logging.Logger to configure console and file handlers.
    """
    @staticmethod
    def get(name: str,
            level: int = logging.INFO,
            log_file: str = None) -> logging.Logger:
        """
        Create and return a configured logger.

        Parameters
        ----------
        name : str
            Logger name.
        level : int
            Logging level (e.g., logging.INFO).
        log_file : str, optional
            Path to a file to write logs; if None, file handler is omitted.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid adding handlers multiple times
        if getattr(logger, '_configured', False):
            return logger
        logger._configured = True

        fmt = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger
