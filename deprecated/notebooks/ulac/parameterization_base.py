from dataclasses import dataclass

import numpy as np
import pyvista as pv


# ==================================================================================================
@dataclass
class ParameterizedPath:
    inds: np.ndarray = None
    relative_length: np.ndarray = None


@dataclass
class UACPath:
    inds: np.ndarray = None
    alpha: np.ndarray = None
    beta: np.ndarray = None


@dataclass
class UACCircle:
    center: tuple[float, float] = (0, 0)
    radius: float = 0
    start_angle: float = 0
    orientation: float = 0


@dataclass
class UACRectangle:
    lower_left_corner: tuple[float, float] = (0, 0)
    length_alpha: float = 0
    length_beta: float = 0

@dataclass
class UACLine:
    start: tuple[float, float] = (0, 0)
    end: tuple[float, float] = (0, 0)


# ==================================================================================================
def parameterize_polyline_by_path_length(
    mesh: pv.PolyData,
    boundary: np.ndarray,
    start_ind: int,
    marker_inds: list[int],
    marker_values: list[float],
) -> np.ndarray:
    ordered_boundary, relative_marker_inds = _reorder_by_start_ind_and_markers(
        boundary, start_ind, marker_inds, marker_values
    )
    segments = np.split(np.arange(ordered_boundary.size), relative_marker_inds)
    segment_values = [0, *marker_values, 1]
    boundary_coordinates = np.array(mesh.points[ordered_boundary])
    path_parameterization = _construct_parameterization(
        boundary_coordinates, segments, segment_values
    )
    parameterized_boundary = ParameterizedPath(
        inds=ordered_boundary, relative_length=path_parameterization
    )

    return parameterized_boundary





# ==================================================================================================
def _reorder_by_start_ind_and_markers(
    boundary: np.ndarray, start_ind: int, marker_inds: list[int]=[], marker_values: list[float]=[]
) -> np.ndarray:
    relative_start_ind_location = np.where(boundary == start_ind)[0][0]
    ordered_boundary = np.roll(boundary, -relative_start_ind_location)

    relative_marker_inds = [np.where(ordered_boundary == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        ordered_boundary = np.roll(np.flip(ordered_boundary), 1)
        relative_marker_inds = [np.where(ordered_boundary == ind)[0][0] for ind in marker_inds]
    return ordered_boundary, relative_marker_inds


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
