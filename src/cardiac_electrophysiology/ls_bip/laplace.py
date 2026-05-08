import numpy as np

from . import components, lowrank


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
    def compute_pointwise_variance(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        prior_variance = sum(
            eigenvalue * self._prior_covariance_eigenvectors[:, i] ** 2
            for i, eigenvalue in enumerate(self._prior_covariance_eigenvalues)
        )
        hessian_variance = sum(
            eigenvalue / (1 + eigenvalue) * self._hessian_eigenvectors[:, i] ** 2
            for i, eigenvalue in enumerate(self._hessian_eigenvalues)
        )
        return prior_variance, hessian_variance
