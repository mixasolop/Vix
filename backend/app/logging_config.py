import logging
import sys


def configure_logging() -> None:
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    existing_console_handler = next(
        (handler for handler in root_logger.handlers if getattr(handler, "_wma_console_handler", False)),
        None,
    )
    if existing_console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler._wma_console_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(console_handler)
    else:
        console_handler = existing_console_handler

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logging.getLogger("app").setLevel(logging.INFO)
