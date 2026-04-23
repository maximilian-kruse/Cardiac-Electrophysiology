import logging
import sys
from dataclasses import dataclass
from pathlib import Path


# ==================================================================================================
@dataclass
class LSOPTLoggerSettings:
    print_to_console: bool = True
    logfile_path: Path | None = None
    write_mode: str = "w"

@dataclass
class LogEntry:
    value: float
    str_id: str
    str_format: str


# ==================================================================================================
class LSOPTLogger:
    def __init__(self, logger_settings: LSOPTLoggerSettings) -> None:
        self._logfile_path = logger_settings.logfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.print_to_console:
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

    def log_header(self, outputs: tuple[LogEntry]) -> None:
        log_header_str = f"| "
        for out in outputs:
            log_header_str += f"{out.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    def log_outputs(self, outputs: tuple[LogEntry]) -> None:
        output_str = "| "
        for out in outputs:
            value_str = f"{out.value:{out.str_format}}"
            output_str += f"{value_str}| "
        self.info(output_str)

    def info(self, message: str) -> None:
        self._pylogger.info(message)

    def debug(self, message: str) -> None:
        self._pylogger.debug(message)

    def exception(self, message: str) -> None:
        self._pylogger.exception(message)

    def error(self, message: str) -> None:
        self._pylogger.error(message)
