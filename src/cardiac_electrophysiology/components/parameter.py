import numpy as np


# ==================================================================================================
class AngleParameterTransformator:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, mean_angle: float, clip_tolerance: float = 1e-6) -> None:
        self.mean_angle = mean_angle
        self._clip_tolerance = clip_tolerance
        self.mean_parameter = self._transform_angle_to_parameter(mean_angle + np.pi / 2)

    # ----------------------------------------------------------------------------------------------
    def compute_parameter_from_angle(
        self, angle: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        centered_angles = angle - self.mean_angle + np.pi / 2
        centered_parameter = self._transform_angle_to_parameter(centered_angles)
        parameter = centered_parameter + self.mean_parameter
        return parameter

    # ----------------------------------------------------------------------------------------------
    def compute_angle_from_parameter(
        self, parameter: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        centered_parameter = parameter - self.mean_parameter
        angle = np.arccos(np.tanh(centered_parameter)) + self.mean_angle - np.pi / 2
        return angle

    # ----------------------------------------------------------------------------------------------
    def _transform_angle_to_parameter(
        self, angle: np.ndarray[tuple[int], np.dtype[np.float64]] | float,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | float:
        parameter = np.arctanh(
            np.cos(np.clip(angle, self._clip_tolerance, 2 * np.pi - self._clip_tolerance))
        )
        return parameter


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
        angle[angle < 0] += np.pi
        return angle
