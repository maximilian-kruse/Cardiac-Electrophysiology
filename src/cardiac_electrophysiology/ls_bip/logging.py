import logging
import sys
from dataclasses import dataclass
from pathlib import Path


# ==================================================================================================
@dataclass
class LoggerSettings:
    do_printing: bool | None = True
    logfile_path: Path | None = None
    write_mode: str | None = "w"


# ==================================================================================================
class LSBIPLogger:
    def __init__(self, logger_settings: LoggerSettings) -> None:
        self._logfile_path = logger_settings.logfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if self._logfile_path is not None:
                self._logfile_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(
                    self._logfile_path, mode=logger_settings.write_mode
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                self._pylogger.addHandler(file_handler)

    def log_header(self, header_message: str) -> None:
        self.info(f"======== {header_message} ========")

    def info(self, message: str) -> None:
        self._pylogger.info(message)

    def debug(self, message: str) -> None:
        self._pylogger.debug(message)

    def exception(self, message: str) -> None:
        self._pylogger.exception(message)

    def error(self, message: str) -> None:
        self._pylogger.error(message)
