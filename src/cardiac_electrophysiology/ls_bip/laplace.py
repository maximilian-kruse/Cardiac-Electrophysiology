from functools import partial

import numpy as np
import scipy.sparse as sps

from . import components, lowrank, posterior


# ==================================================================================================
def compute_preconditioned_hessian_lowrank_approximation(
    posterior: posterior.LogPosterior,
    map_estimate: np.ndarray[tuple[int], np.dtype[np.float64]],
    map_solution: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_eigenvalues: int,
    oversampling_factor: int,
    seed: int | None = None,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]
]:
    observation_matrix = posterior.likelihood.observation_matrix
    precision_matrix = posterior.likelihood.precision_matrix
    jvp_callable = partial(
        posterior.parameter_to_solution_map.evaluate_jacobian_vector_product,
        parameter_vector=map_estimate,
        solution_vector=map_solution,
    )
    vjp_callable = partial(
        posterior.parameter_to_solution_map.evaluate_gradient,
        parameter_vector=map_estimate,
        solution_vector=map_solution,
    )

    def likelihood_hessian_callable(
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = jvp_callable(direction_vector=direction_vector)
        result = observation_matrix.T @ precision_matrix @ observation_matrix @ result
        result = vjp_callable(adjoint_vector=result)
        return result

    eigenvalues, eigenvectors = lowrank.compute_randomized_generalized_ev_decomposition(
        matrix_callable=likelihood_hessian_callable,
        preconditioner_callable=posterior.prior.evaluate_hessian_vector_product,
        inverse_preconditioner_callable=posterior.prior.apply_covariance_operator,
        fullspace_size=map_estimate.size,
        num_eigenvalues=num_eigenvalues,
        over_sampling_factor=oversampling_factor,
        seed=seed,
    )
    return eigenvalues, eigenvectors


# --------------------------------------------------------------------------------------------------
def compute_prior_covariance_lowrank_approximation(
    prior: components.Prior,
    map_estimate: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_eigenvalues: int,
    oversampling_factor: int,
    seed: int | None = None,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]
]:
    eigenvalues, eigenvectors = lowrank.compute_randomized_generalized_ev_decomposition(
        matrix_callable=prior.apply_covariance_operator,
        preconditioner_callable=lambda v: v,
        inverse_preconditioner_callable=lambda v: v,
        fullspace_size=map_estimate.size,
        num_eigenvalues=num_eigenvalues,
        over_sampling_factor=oversampling_factor,
        seed=seed,
    )
    return eigenvalues, eigenvectors


# ==================================================================================================
class LaplaceApproximation:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        map_estimate: np.ndarray[tuple[int], np.dtype[np.float64]],
        hessian_eigenvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
        hessian_eigenvectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        prior_covariance_eigenvalues: np.ndarray[tuple[int], np.dtype[np.float64]],
        prior_covariance_eigenvectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        prior: components.Prior,
    ):
        self._hessian_operator = lowrank.LowRankOperator(
            eigenvalues=hessian_eigenvalues, eigenvectors=hessian_eigenvectors
        )
        self._hessian_inv_operator = lowrank.LowRankOperator(
            eigenvalues=hessian_eigenvalues / (hessian_eigenvalues + 1),
            eigenvectors=hessian_eigenvectors,
        )
        self._hessian_sqrt_inv_operator = lowrank.LowRankOperator(
            eigenvalues=1 - 1 / np.sqrt(hessian_eigenvalues + 1), eigenvectors=hessian_eigenvectors
        )
        self._map_estimate = map_estimate
        self._hessian_eigenvalues = hessian_eigenvalues
        self._hessian_eigenvectors = hessian_eigenvectors
        self._prior_covariance_eigenvalues = prior_covariance_eigenvalues
        self._prior_covariance_eigenvectors = prior_covariance_eigenvectors
        self._prior = prior

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float | tuple[float, float]:
        diff_vector = parameter_vector - self._map_estimate
        cost = 0.5 * diff_vector.T @ self.apply_precision(diff_vector)
        return cost

    # ----------------------------------------------------------------------------------------------
    def apply_precision(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        tmp_vec = self._prior.evaluate_hessian_vector_product(input_vector)
        result = tmp_vec + self._prior.evaluate_hessian_vector_product(
            self._hessian_operator.apply(tmp_vec)
        )
        return result

    # ----------------------------------------------------------------------------------------------
    def apply_covariance(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = self._prior.apply_covariance_operator(
            input_vector
        ) - self._hessian_inv_operator.apply(input_vector)
        return result

    # ----------------------------------------------------------------------------------------------
    def apply_sampling_factor(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        prior_transformed_vec = self._prior.apply_covariance_factorization(input_vector)
        result = prior_transformed_vec - self._hessian_sqrt_inv_operator.apply(
            self._prior.evaluate_hessian_vector_product(prior_transformed_vec)
        )
        return result

    # ----------------------------------------------------------------------------------------------
    def compute_pointwise_variance(self, return_prior: bool | None = False):
        prior_variance = sum(
            eigenvalue * self._prior_covariance_eigenvectors[:, i] ** 2
            for i, eigenvalue in enumerate(self._prior_covariance_eigenvalues)
        )
        hessian_variance = sum(
            eigenvalue / (1 + eigenvalue) * self._hessian_eigenvectors[:, i] ** 2
            for i, eigenvalue in enumerate(self._hessian_eigenvalues)
        )
        laplace_variance = prior_variance - hessian_variance
        if return_prior:
            return laplace_variance, prior_variance
        else:
            return laplace_variance
