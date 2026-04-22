import numpy as np


# ==================================================================================================
class AngleFiberTransformator:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        basis_vector_one: np.ndarray[tuple[int], np.dtype[np.float64]],
        basis_vector_two: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        self._basis_vector_one = basis_vector_one
        self._basis_vector_two = basis_vector_two

    # ----------------------------------------------------------------------------------------------
    def compute_fiber_from_angle(
        self, angle: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        fiber_vector = (
            np.cos(angle)[:, np.newaxis] * self._basis_vector_one
            + np.sin(angle)[:, np.newaxis] * self._basis_vector_two
        )
        return fiber_vector

    # ----------------------------------------------------------------------------------------------
    def compute_angle_from_fiber(
        self, fiber_vector: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        angle = np.arctan2(
            np.einsum("ij,ij->i", fiber_vector, self._basis_vector_two),
            np.einsum("ij,ij->i", fiber_vector, self._basis_vector_one),
        )
        return angle
