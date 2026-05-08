from collections.abc import Callable
from functools import partial

import numpy as np
import scipy.sparse.linalg as sla

from . import posterior


# ==================================================================================================
class LowRankOperator:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        eigenvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
        eigenvectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self.eigenvectors @ (self.eigenvalues * (self.eigenvectors.T @ vector))

    # ----------------------------------------------------------------------------------------------
    def compute_diagonal(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.sum(self.eigenvalues * self.eigenvectors**2, axis=1)


# ==================================================================================================
class RandomizedDecomposer:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        posterior: posterior.LogPosterior,
        parameter: np.ndarray[tuple[int], np.dtype[np.float64]],
        seed: int | None = 0,
    ):
        self._prior = posterior.prior
        self._observation_matrix = posterior.likelihood.observation_matrix
        self._precision_matrix = posterior.likelihood.precision_matrix
        solution = posterior.parameter_to_solution_map.evaluate_forward(parameter)
        self._jvp_callable = partial(
            posterior.parameter_to_solution_map.evaluate_jacobian_vector_product,
            parameter_vector=parameter,
            solution_vector=solution,
        )
        self._vjp_callable = partial(
            posterior.parameter_to_solution_map.evaluate_gradient,
            parameter_vector=parameter,
            solution_vector=solution,
        )
        self._rng = np.random.default_rng(seed)
        self._fullspace_size = parameter.size

    # ----------------------------------------------------------------------------------------------
    def compute_likelihood_hessian_lowrank_approximation(
        self, num_eigenvalues: int, scaling_factor: float = 1.0, offset_factor: float = 0.0
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        hessian_callable = partial(
            self._likelihood_hessian_callable,
            scaling_factor=scaling_factor,
            offset_factor=offset_factor,
        )
        hessian_operator = sla.LinearOperator(
            shape=(self._fullspace_size, self._fullspace_size),
            matvec=hessian_callable,
        )
        preconditioner_operator = sla.LinearOperator(
            shape=(self._fullspace_size, self._fullspace_size),
            matvec=self._prior_precision_callable,
        )
        inverse_preconditioner_operator = sla.LinearOperator(
            shape=(self._fullspace_size, self._fullspace_size),
            matvec=self._prior_covariance_callable,
        )
        eigenvalues, eigenvectors = sla.eigsh(
            hessian_operator,
            k=num_eigenvalues,
            M=preconditioner_operator,
            Minv=inverse_preconditioner_operator,
            rng=self._rng,
            which="LA",
            tol=1e-6,
            return_eigenvectors=True,
        )
        return eigenvalues[::-1], eigenvectors[:, ::-1]

    # ----------------------------------------------------------------------------------------------
    def compute_prior_covariance_lowrank_approximation(
        self, num_eigenvalues: int
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        covariance_operator = sla.LinearOperator(
            shape=(self._fullspace_size, self._fullspace_size),
            matvec=self._prior_covariance_callable,
        )
        eigenvalues, eigenvectors = sla.eigsh(
            covariance_operator,
            k=num_eigenvalues,
            which="LA",
            tol=1e-6,
            rng=self._rng,
            return_eigenvectors=True,
        )
        return eigenvalues[::-1], eigenvectors[:, ::-1]

    # ----------------------------------------------------------------------------------------------
    def _likelihood_hessian_callable(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        scaling_factor: float,
        offset_factor: float,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        direction_vector = direction_vector.astype(np.float64)
        result = self._jvp_callable(direction_vector=direction_vector)
        result = (
            self._observation_matrix.T @ self._precision_matrix @ self._observation_matrix @ result
        )
        result = self._vjp_callable(adjoint_vector=result)
        return scaling_factor * result + offset_factor

    # ----------------------------------------------------------------------------------------------
    def _prior_covariance_callable(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        direction_vector = direction_vector.astype(np.float64)
        result = self._prior.apply_covariance_operator(direction_vector)
        return result

    # ----------------------------------------------------------------------------------------------
    def _prior_precision_callable(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        direction_vector = direction_vector.astype(np.float64)
        result = self._prior.evaluate_hessian_vector_product(direction_vector)
        return result


# ==================================================================================================
# def _compute_preconditioned_qr_decomposition_mgs(
#     matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
#     preconditioner_callable: Callable[
#         [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
#     ],
# ) -> tuple[
#     np.ndarray[tuple[int, int], np.dtype[np.float64]],
#     np.ndarray[tuple[int, int], np.dtype[np.float64]],
# ]:
#     num_rows, num_cols = matrix.shape
#     Q = np.zeros((num_rows, num_cols), dtype=matrix.dtype)
#     R = np.zeros((num_cols, num_cols), dtype=matrix.dtype)

#     for j in range(num_cols):
#         v = matrix[:, j].copy()
#         for i in range(j):
#             q_i = Q[:, i]
#             R[i, j] = np.vdot(q_i, preconditioner_callable(v))
#             v -= R[i, j] * q_i
#         R[j, j] = np.sqrt(np.vdot(v, preconditioner_callable(v)))
#         Q[:, j] = v / R[j, j]

#     return Q, R


# def compute_preconditioned_qr_decomposition_mgs(matrix, preconditioner):
#     n, k = matrix.shape
#     q = matrix.astype(float, copy=True)
#     Aq = np.empty_like(q)
#     r = np.zeros((k, k))

#     # --- First column ---
#     Aq[:, 0] = preconditioner(q[:, 0])
#     r[0, 0] = np.sqrt(q[:, 0] @ Aq[:, 0])
#     q[:, 0] /= r[0, 0]
#     Aq[:, 0] /= r[0, 0]

#     # --- Remaining columns ---
#     for j in range(1, k):
#         Aq[:, j] = preconditioner(q[:, j])
#         for i in range(j):
#             r[i, j] = q[:, j] @ Aq[:, i]
#             q[:, j] -= r[i, j] * q[:, i]
#             Aq[:, j] -= r[i, j] * Aq[:, i]
#         r[j, j] = np.sqrt(q[:, j] @ Aq[:, j])

#         if abs(r[j, j]) < 1e-14:
#             k = j
#             print("A-orthonormalization broke down")
#             break

#         q[:, j] /= r[j, j]
#         Aq[:, j] /= r[j, j]

#     q, Aq, r = q[:, :k], Aq[:, :k], r[:k, :k]
#     return q, r


# # ==================================================================================================
# def compute_prechol_qr(matrix, preconditioner_callable):
#     Z, R1 = np.linalg.qr(matrix)
#     Z_precon = np.zeros_like(Z)
#     for i in range(Z.shape[1]):
#         Z_precon[:, i] = preconditioner_callable(Z[:, i])
#     R2 = np.linalg.cholesky(Z.T @ Z_precon)
#     Q = Z @ np.linalg.inv(R2)
#     R = R2 @ R1
#     return Q, R


# # ==================================================================================================
# def compute_randomized_generalized_ev_decomposition(
#     matrix_callable: Callable[
#         [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
#     ],
#     preconditioner_callable: Callable[
#         [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
#     ],
#     inverse_preconditioner_callable: Callable[
#         [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
#     ],
#     fullspace_size: int,
#     num_eigenvalues: int,
#     over_sampling_factor: int,
#     seed: int | None = 0,
# ) -> tuple[
#     np.ndarray[tuple[int], np.dtype[np.float64]],
#     np.ndarray[tuple[int, int], np.dtype[np.float64]],
# ]:
#     rng = np.random.default_rng(seed)
#     random_gaussian_matrix = rng.standard_normal(
#         size=(fullspace_size, num_eigenvalues + over_sampling_factor), dtype=np.float64
#     )
#     matrix_to_factorize = np.zeros_like(random_gaussian_matrix)
#     for i in range(num_eigenvalues + over_sampling_factor):
#         matrix_to_factorize[:, i] = inverse_preconditioner_callable(
#             matrix_callable(random_gaussian_matrix[:, i])
#         )
#     Q, _ = compute_prechol_qr(matrix_to_factorize, preconditioner_callable)
#     matrix_for_evd = np.zeros(
#         (num_eigenvalues + over_sampling_factor, num_eigenvalues + over_sampling_factor),
#         dtype=np.float64,
#     )
#     for i in range(num_eigenvalues + over_sampling_factor):
#         matrix_for_evd[:, i] = Q.T @ matrix_callable(Q[:, i])
#     eigenvalues, eigenvectors = np.linalg.eigh(matrix_for_evd)
#     sorted_eigenvalues = eigenvalues[::-1]
#     sorted_eigenvectors = eigenvectors[:, ::-1]
#     return sorted_eigenvalues[:num_eigenvalues], Q @ sorted_eigenvectors[:, :num_eigenvalues]
