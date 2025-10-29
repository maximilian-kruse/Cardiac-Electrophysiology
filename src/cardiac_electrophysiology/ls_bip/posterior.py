import numpy as np

from . import components


# ==================================================================================================
class CachedState:
    # ----------------------------------------------------------------------------------------------
    def __init__(self):
        self.parameter_vector = None
        self._solution_vector = None
        self._gradient_vector = None

    # ----------------------------------------------------------------------------------------------
    def set_solution_vector(
        self,
        value: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self.parameter_vector is None or not np.allclose(
            parameter_vector, self.parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._solution_vector = value

    # ----------------------------------------------------------------------------------------------
    def get_solution_vector(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self.parameter_vector is None or not np.allclose(
            parameter_vector, self.parameter_vector
        ):
            raise ValueError(
                "Cached solution vector is not available for the given parameter vector."
            )
        return self._solution_vector

    # ----------------------------------------------------------------------------------------------
    def set_gradient_vector(
        self,
        value: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self.parameter_vector is None or not np.allclose(
            parameter_vector, self.parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._gradient_vector = value

    # ----------------------------------------------------------------------------------------------
    def get_gradient_vector(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self.parameter_vector is None or not np.allclose(
            parameter_vector, self.parameter_vector
        ):
            raise ValueError(
                "Cached gradient vector is not available for the given parameter vector."
            )
        return self._gradient_vector


# ==================================================================================================
class LogPosterior:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        likelihood: object,
        parameter_to_solution_map: components.ParameterToSolutionMap,
        prior: object,
    ):
        self._likelihood = likelihood
        self._parameter_to_solution_map = parameter_to_solution_map
        self._prior = prior
        self._cached_state = CachedState()

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        solution_vector = self._parameter_to_solution_map.evaluate_forward(parameter_vector)
        likelihood_cost = self._likelihood.evaluate_cost(solution_vector)
        prior_cost = self._prior.evaluate_cost(parameter_vector)
        total_cost = likelihood_cost + prior_cost
        self._cached_state.parameter_vector = parameter_vector
        self._cached_state.set_solution_vector(solution_vector, parameter_vector)
        return total_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        solution_vector = self._cached_state.get_solution_vector(parameter_vector)
        likelihood_gradient = self._likelihood.evaluate_gradient(solution_vector)
        pts_gradient = self._parameter_to_solution_map.evaluate_gradient(
            solution_vector, parameter_vector, likelihood_gradient
        )
        prior_gradient = self._prior.evaluate_gradient(parameter_vector)
        total_gradient = pts_gradient + prior_gradient
        self._cached_state.set_gradient_vector(pts_gradient, parameter_vector)
        return total_gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError
