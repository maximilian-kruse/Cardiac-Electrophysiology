import numpy as np

from . import components


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

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        solution_vector = self._parameter_to_solution_map.evaluate_forward(parameter_vector)
        likelihood_cost = self._likelihood.evaluate_cost(solution_vector)
        prior_cost = self._prior.evaluate_cost(parameter_vector)
        total_cost = likelihood_cost + prior_cost
        return total_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        likelihood_gradient = self._likelihood.evaluate_gradient(solution_vector)
        pts_gradient = self._parameter_to_solution_map.evaluate_gradient(
            parameter_vector, adjoint_vector=likelihood_gradient
        )
        prior_gradient = self._prior.evaluate_gradient(parameter_vector)
        total_gradient = pts_gradient + prior_gradient
        return total_gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError
