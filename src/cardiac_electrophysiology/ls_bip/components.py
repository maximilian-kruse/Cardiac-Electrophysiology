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
        difference_vector = self._observation_matrix @ solution_vector - self._data_vector
        cost = 0.5 * difference_vector.T @ self._precision_matrix @ difference_vector
        return cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        difference_vector = self._observation_matrix @ solution_vector - self._data_vector
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
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError
