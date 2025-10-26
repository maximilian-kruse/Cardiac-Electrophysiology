from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


# ==================================================================================================
def assemble_vertex_observation_matrix(
    num_vertices: int, observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.int64]]
) -> sp.coo_matrix:
    row_inds = np.arange(len(observed_vertex_indices), dtype=np.int64)
    col_inds = observed_vertex_indices
    data = np.ones(len(observed_vertex_indices), dtype=np.float64)
    observation_matrix = sp.coo_matrix(
        (data, (row_inds, col_inds)), shape=(len(observed_vertex_indices), num_vertices)
    )
    return observation_matrix


# --------------------------------------------------------------------------------------------------
def assemble_diagonal_precision_matrix(
    precision_values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> sp.coo_matrix:
    num_observations = precision_values.shape[0]
    row_inds = np.arange(num_observations, dtype=np.int64)
    col_inds = np.arange(num_observations, dtype=np.int64)
    data = precision_values
    precision_matrix = sp.coo_matrix(
        (data, (row_inds, col_inds)), shape=(num_observations, num_observations)
    )
    return precision_matrix


# ==================================================================================================
class ParameterToSolutionMap(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_forward(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        pass

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError


# ==================================================================================================
class GaussianLogLikelihood:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        data_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        observation_matrix: sp.coo_matrix,
        precision_matrix: sp.coo_matrix,
    ):
        self._data_vector = data_vector
        self._observation_matrix = observation_matrix
        self._precision_matrix = precision_matrix

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        difference_vector = self._observation_matrix @ (solution_vector - self._data_vector)
        cost = 0.5 * difference_vector.T @ self._precision_matrix @ difference_vector
        return cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        difference_vector = self._observation_matrix @ (solution_vector - self._data_vector)
        gradient = self._observation_matrix.T @ self._precision_matrix @ difference_vector
        return gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        hvp = (
            self._observation_matrix.T
            @ self._precision_matrix
            @ self._observation_matrix
            @ direction_vector
        )
        return hvp


# ==================================================================================================
class LogPosterior:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        likelihood: object,
        parameter_to_solution_map: ParameterToSolutionMap,
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
