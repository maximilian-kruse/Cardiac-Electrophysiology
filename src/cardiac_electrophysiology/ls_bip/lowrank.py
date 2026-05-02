from collections.abc import Callable

import numpy as np


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


# ==================================================================================================
def _compute_preconditioned_qr_decomposition_mgs(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    preconditioner_callable: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ],
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
]:
    num_rows, num_cols = matrix.shape
    Q = np.zeros((num_rows, num_cols), dtype=matrix.dtype)
    R = np.zeros((num_cols, num_cols), dtype=matrix.dtype)

    for j in range(num_cols):
        v = matrix[:, j].copy()
        for i in range(j):
            q_i = Q[:, i]
            R[i, j] = np.vdot(q_i, preconditioner_callable(v))
            v -= R[i, j] * q_i
        R[j, j] = np.sqrt(np.vdot(v, preconditioner_callable(v)))
        Q[:, j] = v / R[j, j]

    return Q, R


def compute_preconditioned_qr_decomposition_mgs(matrix, preconditioner):
    n, k = matrix.shape
    q  = matrix.astype(float, copy=True)
    Aq = np.empty_like(q)
    r  = np.zeros((k, k))

    # --- First column ---
    Aq[:, 0] = preconditioner(q[:, 0])
    r[0, 0]  = np.sqrt(q[:, 0] @ Aq[:, 0])
    q[:, 0]  /= r[0, 0]
    Aq[:, 0] /= r[0, 0]

    # --- Remaining columns ---
    for j in range(1, k):
        Aq[:, j] = preconditioner(q[:, j])
        for i in range(j):
            r[i, j]   = q[:, j] @ Aq[:, i]
            q[:, j]  -= r[i, j] * q[:, i]
            Aq[:, j] -= r[i, j] * Aq[:, i]
        r[j, j] = np.sqrt(q[:, j] @ Aq[:, j])

        if abs(r[j, j]) < 1e-14:
            k = j
            print("A-orthonormalization broke down")
            break

        q[:, j]  /= r[j, j]
        Aq[:, j] /= r[j, j]

    q, Aq, r = q[:, :k], Aq[:, :k], r[:k, :k]
    return q, r


# ==================================================================================================
def compute_randomized_generalized_ev_decomposition(
    matrix_callable: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ],
    preconditioner_callable: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ],
    inverse_preconditioner_callable: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ],
    fullspace_size: int,
    num_eigenvalues: int,
    over_sampling_factor: int,
    seed: int | None = 0,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
]:
    rng = np.random.default_rng(seed)
    random_gaussian_matrix = rng.standard_normal(
        size=(fullspace_size, num_eigenvalues + over_sampling_factor), dtype=np.float64
    )
    matrix_to_factorize = np.zeros_like(random_gaussian_matrix)
    for i in range(num_eigenvalues + over_sampling_factor):
        matrix_to_factorize[:, i] = inverse_preconditioner_callable(
            matrix_callable(random_gaussian_matrix[:, i])
        )
    Q, _ = compute_preconditioned_qr_decomposition_mgs(matrix_to_factorize, preconditioner_callable)
    matrix_for_evd = np.zeros(
        (num_eigenvalues + over_sampling_factor, num_eigenvalues + over_sampling_factor),
        dtype=np.float64,
    )
    for i in range(num_eigenvalues + over_sampling_factor):
        matrix_for_evd[:, i] = Q.T @ matrix_callable(Q[:, i])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_for_evd)
    sorted_eigenvalues = eigenvalues[::-1]
    sorted_eigenvectors = eigenvectors[:, ::-1]
    return sorted_eigenvalues[:num_eigenvalues], Q @ sorted_eigenvectors[:, :num_eigenvalues]
