import sys
import json
import functools
import logging
from binance.exceptions import BinanceAPIException
from utils.logger_utils import log

logger = logging.getLogger(__name__)


def exception_handler(default_return=None):
    """
    A decorator that catches exceptions, logs the error, optionally rolls back the database session,
    and sends an email notification.

    Args:
        default_return (Any, optional): The value to return if an exception occurs. Defaults to None.
        db_rollback (bool, optional): If True, rolls back the database session upon an exception. Defaults to False.

    Returns:
        function: A wrapped function that handles exceptions.

    Exceptions Caught:
        - IndexError
        - BinanceAPIException
        - ConnectionError
        - TimeoutError
        - ValueError
        - TypeError
        - FileNotFoundError
        - General Exception (any other unexpected errors)

    Behavior:
        - Logs the exception with the bot ID (if available).
        - Sends an email notification to the administrator.
        - Optionally rolls back the database session if `db_rollback=True`.
        - Returns `default_return` in case of an error.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (
                IndexError,
                BinanceAPIException,
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
                FileNotFoundError,
                json.JSONDecodeError,
            ) as e:
                exception_type = type(e).__name__
                log(f"{exception_type}: {e}")
            except Exception as e:
                exception_type = "Exception"
                log(f"{exception_type}: {e}")

            if default_return is exit:
                log("Exiting program due to an error.")
                sys.exit(1)
            elif callable(default_return):
                return default_return()
            return default_return

        return wrapper

    return decorator
