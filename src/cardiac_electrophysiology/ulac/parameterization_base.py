from dataclasses import dataclass

import numpy as np
import pyvista as pv


# ==================================================================================================
@dataclass
class ParameterizedPath:
    points: np.ndarray=None
    values: np.ndarray=None


# ==================================================================================================
def parameterize_boundary_by_path_length(
    mesh: pv.PolyData,
    boundary: np.ndarray,
    start_ind: int,
    marker_inds: list[int],
    marker_values: list[float],
) -> np.ndarray:
    ordered_boundary = _reorder_by_start_ind(boundary, start_ind)
    ordered_boundary, relative_marker_inds = _reorder_by_markers(
        ordered_boundary, marker_inds, marker_values
    )
    segments = _construct_segments(ordered_boundary.size, relative_marker_inds)

    segment_values = [0, *marker_values, 1]
    boundary_coordinates = np.array(mesh.points[ordered_boundary])
    path_parameterization = _construct_parameterization(
        boundary_coordinates, segments, segment_values
    )
    parameterized_boundary = ParameterizedPath(
        points=ordered_boundary, values=path_parameterization
    )

    return parameterized_boundary


# --------------------------------------------------------------------------------------------------
def _reorder_by_start_ind(boundary: np.ndarray, start_ind: int) -> np.ndarray:
    relative_start_ind_location = np.where(boundary == start_ind)[0][0]
    ordered_boundary = np.concatenate(
        [boundary[relative_start_ind_location:], boundary[:relative_start_ind_location]]
    )
    return ordered_boundary


# --------------------------------------------------------------------------------------------------
def _reorder_by_markers(
    boundary: np.ndarray, marker_inds: list[int], marker_values: list[float]
) -> np.ndarray:
    relative_marker_inds = [np.where(boundary == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        boundary = np.roll(np.flip(boundary), 1)
        relative_marker_inds = [np.where(boundary == ind)[0][0] for ind in marker_inds]
    return boundary, relative_marker_inds


# --------------------------------------------------------------------------------------------------
def _construct_segments(
    num_boundary_points: int,
    relative_marker_inds: list[int],
) -> np.ndarray:
    relative_boundary_inds = np.arange(num_boundary_points)
    segments = [relative_boundary_inds[0 : relative_marker_inds[0]]]
    for i in range(len(relative_marker_inds) - 1):
        segments.append(  # noqa: PERF401
            relative_boundary_inds[relative_marker_inds[i] : relative_marker_inds[i + 1]]
        )
    segments.append(relative_boundary_inds[relative_marker_inds[-1] :])
    return segments


# --------------------------------------------------------------------------------------------------
def _construct_parameterization(
    boundary_coordinates: np.ndarray, segments: list[np.ndarray], segment_values: list[float]
) -> list[np.ndarray]:
    edge_lengths = np.linalg.norm(boundary_coordinates[1:] - boundary_coordinates[:-1], axis=1)
    relative_path_length = np.zeros(boundary_coordinates.shape[0])
    relative_path_length[1:] = np.cumsum(edge_lengths)
    relative_path_length /= relative_path_length[-1]

    path_parameterization = np.zeros(boundary_coordinates.shape[0])
    for i, segment_inds in enumerate(segments):
        start_value = segment_values[i]
        end_value = segment_values[i + 1]
        num_points = segment_inds.size
        segment_parameterization = np.linspace(start_value, end_value, num_points, endpoint=False)
        path_parameterization[segment_inds] = segment_parameterization

    return path_parameterization
