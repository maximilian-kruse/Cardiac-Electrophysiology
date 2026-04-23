import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import scipy.optimize as spo

from . import logging


# ==================================================================================================
class BaseConfig(ABC):
    pass


@dataclass
class OptimizationResult:
    result: np.ndarray[tuple[int], np.dtype[np.float64]]
    loss_history: list[float]
    gradient_norm_history: list[float]
    num_iterations: int
    success: bool
    status_message: str


# ==================================================================================================
class BaseOptimizer(ABC):
    requires_hessian: bool | None = None

    # ----------------------------------------------------------------------------------------------
    def __init__(self, config: BaseConfig, logger_settings: logging.LSOPTLoggerSettings) -> None:
        self._config = config
        self._logger = logging.LSOPTLogger(logger_settings)
        self.loss_history = None
        self.gradient_norm_history = None
        self._start_time = None
        self._log_outputs = None
        self._iteration = None

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        loss_function: Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float],
        gradient_function: Callable[
            [np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        hvp_function: Callable[
            [np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ]
        | None = None,
    ) -> OptimizationResult:
        if self.requires_hessian and hvp_function is None:
            raise ValueError(
                "This optimizer requires a Hessian or Hessian-vector product function."
            )
        wrapped_loss_function = store(self, "loss")(loss_function)
        wrapped_gradient_function = store(self, "grad")(gradient_function)

        self.loss_history = []
        self.gradient_norm_history = []
        self._start_time = time.time()
        self._iteration = 0
        self._log_outputs = self._set_up_logging_output()
        self._logger.log_header(tuple(self._log_outputs.values()))
        result = self._run_impl(
            initial_guess,
            wrapped_loss_function,
            wrapped_gradient_function,
            hvp_function,
            self._callback,
        )
        return self._create_optimization_result(result)

    # ----------------------------------------------------------------------------------------------
    def _set_up_logging_output(self) -> None:
        logging_outputs = {
            "iteration": logging.LogEntry(
                value=np.inf, str_id=f"{'Iteration':<12}", str_format="<+12.3e"
            ),
            "time": logging.LogEntry(value=np.inf, str_id=f"{'Time':<12}", str_format="<+12.3e"),
            "loss": logging.LogEntry(value=np.inf, str_id=f"{'Loss':<12}", str_format="<+12.3e"),
            "grad_norm": logging.LogEntry(
                value=np.inf, str_id=f"{'Grad Norm':<12}", str_format="<+12.3e"
            ),
        }
        return logging_outputs

    # ----------------------------------------------------------------------------------------------
    def _callback(self, *_: Any) -> None:
        self._iteration += 1
        current_time = time.time() - self._start_time
        self._log_outputs["iteration"].value = self._iteration
        self._log_outputs["time"].value = current_time
        self._log_outputs["loss"].value = self.loss_history[-1]
        self._log_outputs["grad_norm"].value = self.gradient_norm_history[-1]
        self._logger.log_outputs(tuple(self._log_outputs.values()))

    # ----------------------------------------------------------------------------------------------
    def _create_optimization_result(self, result: Any) -> OptimizationResult:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _run_impl() -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError


# ==================================================================================================
def store(optimizer: BaseOptimizer, target: str) -> Callable[[callable], callable]:
    def decorator(function: callable) -> callable:
        @wraps(function)
        def wrapper(
            argument: np.ndarray[tuple[int], np.dtype[np.float64]],
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
            output = function(argument)
            if target == "loss":
                optimizer.loss_history.append(float(output))
            elif target == "grad":
                optimizer.gradient_norm_history.append(float(np.linalg.norm(output)))
            return output

        return wrapper

    return decorator


# ==================================================================================================
@dataclass
class LBFGSConfig(BaseConfig):
    maximum_num_iterations: int
    relative_function_tolerance: float
    relative_gradient_tolerance: float
    max_line_search_steps: int


class LBFGSOptimizer(BaseOptimizer):
    requires_hessian = False

    # ----------------------------------------------------------------------------------------------
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        loss_function: Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float],
        gradient_function: Callable[
            [np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        _hvp_function: Callable[
            [np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ]
        | None = None,
        callback: Callable[[], None] | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

        optimizer_options = {
            "maxiter": self._config.maximum_num_iterations,
            "ftol": self._config.relative_function_tolerance,
            "gtol": self._config.relative_gradient_tolerance,
            "maxls": self._config.max_line_search_steps,
        }
        result = spo.minimize(
            fun=loss_function,
            x0=initial_guess,
            jac=gradient_function,
            method="L-BFGS-B",
            callback=callback,
            options=optimizer_options,
        )
        return result

    # ----------------------------------------------------------------------------------------------
    def _create_optimization_result(self, result: spo.OptimizeResult) -> OptimizationResult:
        return OptimizationResult(
            result=result.x,
            loss_history=self.loss_history,
            gradient_norm_history=self.gradient_norm_history,
            num_iterations=result.nit,
            success=result.success,
            status_message=result.message,
        )
