from dataclasses import dataclass

import igl
import numpy as np
import pyvista as pv
import trimesh as tm


# ==================================================================================================
@dataclass
class ParameterizedPath:
    inds: np.ndarray = None
    relative_lengths: np.ndarray = None


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
def get_inner_outer_boundaries(
    mesh: pv.PolyData, feature_tag: int
) -> tuple[np.ndarray, np.ndarray]:
    tm_mesh = tm.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    feature_mesh = mesh.extract_values(feature_tag, scalars="anatomical_tags")
    feature_boundaries = feature_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    geometry_boundaries = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    feature_boundary_inds = feature_mesh.point_data["vtkOriginalPointIds"][
        feature_boundaries.point_data["vtkOriginalPointIds"]
    ]
    geometry_boundary_inds = geometry_boundaries.point_data["vtkOriginalPointIds"]
    inner_feature_boundary_inds = np.setdiff1d(feature_boundary_inds, geometry_boundary_inds)
    outer_feature_boundary_inds = np.intersect1d(feature_boundary_inds, geometry_boundary_inds)

    boundary_paths = []
    for boundary_inds in [inner_feature_boundary_inds, outer_feature_boundary_inds]:
        if boundary_inds.size > 0:
            ordered_boundary_inds = _construct_ordered_path_from_indices(tm_mesh, boundary_inds)
            boundary_paths.append(ordered_boundary_inds[:-1])
        else:
            boundary_paths.append(np.array([], dtype=int))

    return boundary_paths


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData, subset_one: np.ndarray, subset_two: np.ndarray
):
    vertices = np.array(mesh.points)
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:])

    distances_one = igl.exact_geodesic(V=vertices, F=simplices, VS=subset_one, VT=subset_two)
    end_point = subset_two[np.argmin(distances_one)]
    distances_two = igl.exact_geodesic(
        V=vertices, F=simplices, VS=np.array([end_point]), VT=subset_one
    )
    start_point = subset_one[np.argmin(distances_two)]
    optimal_path = mesh.geodesic(start_point, end_point).point_data["vtkOriginalPointIds"]
    return np.array(optimal_path, dtype=int)


# --------------------------------------------------------------------------------------------------
def _construct_ordered_path_from_indices(
    tm_mesh: tm.Trimesh, path_indices: np.ndarray
) -> np.ndarray:
    path_edges = tm_mesh.edges[np.isin(tm_mesh.edges, path_indices).all(axis=1)].flatten()
    local_edges = np.array([np.where(path_indices == ind)[0][0] for ind in path_edges])
    local_edges = local_edges.reshape(-1, 2)
    path_dict = tm.path.exchange.misc.edges_to_path(local_edges, tm_mesh.vertices[path_indices])
    ordered_vertices = path_dict["entities"][0].discrete(tm_mesh.vertices[path_indices])
    ordered_path_indices = np.array(
        [
            np.where(np.isclose(tm_mesh.vertices, point).all(axis=1))[0][0]
            for point in ordered_vertices
        ]
    )
    return ordered_path_indices


# ==================================================================================================
def reorder_path_by_markers(
    path: np.ndarray,
    start_ind: int,
    marker_inds: list[int]=[],  # noqa: B006
    marker_values: list[float]=[],  # noqa: B006
) -> tuple[np.ndarray, list[int]]:
    relative_start_ind_location = np.where(path == start_ind)[0]
    ordered_path = np.roll(path, -relative_start_ind_location)

    relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        ordered_path = np.roll(np.flip(ordered_path), 1)
        relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    return ordered_path, relative_marker_inds


# --------------------------------------------------------------------------------------------------
def parameterize_path_by_relative_length(
    mesh: pv.PolyData,
    path: np.ndarray,
    relative_marker_inds: list[int]=[],  # noqa: B006
    marker_values: list[float]=[],  # noqa: B006
) -> ParameterizedPath:
    segments = np.split(np.arange(path.size), relative_marker_inds)
    segment_values = [0, *marker_values, 1]
    coordinates = np.array(mesh.points[path])
    edge_lengths = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    relative_path_length = np.zeros(coordinates.shape[0])
    relative_path_length[1:] = np.cumsum(edge_lengths)
    relative_path_length /= relative_path_length[-1]

    relative_lengths = np.zeros(coordinates.shape[0])
    for i, segment_inds in enumerate(segments):
        start_value = segment_values[i]
        end_value = segment_values[i + 1]
        num_points = segment_inds.size
        segment_parameterization = np.linspace(start_value, end_value, num_points, endpoint=False)
        relative_lengths[segment_inds] = segment_parameterization

    parameterized_path = ParameterizedPath(path, relative_lengths)
    return parameterized_path


# ==================================================================================================
def compute_uacs_circle(path: par.ParameterizedPath, uac_circle: UACCircle) -> UACPath:
    angle = uac_circle.start_angle + 2 * np.pi * uac_circle.orientation * path.relative_lengths
    alpha = uac_circle.center[0] + uac_circle.radius * np.cos(angle)
    beta = uac_circle.center[1] + uac_circle.radius * np.sin(angle)
    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path


# --------------------------------------------------------------------------------------------------
def compute_uacs_rectangle(path: par.ParameterizedPath, uac_rectangle: UACRectangle) -> UACPath:
    first_edge = np.where(path.relative_lengths <= 0.25)[0]
    second_edge = np.where((path.relative_lengths > 0.25) & (path.relative_lengths < 0.5))[0]
    third_edge = np.where((path.relative_lengths >= 0.5) & (path.relative_lengths < 0.75))[0]
    fourth_edge = np.where(path.relative_lengths >= 0.75)[0]
    alpha = np.zeros(path.relative_lengths.size)
    beta = np.zeros(path.relative_lengths.size)

    alpha[first_edge] = 4 * uac_rectangle.length_alpha * path.relative_lengths[first_edge]
    alpha[second_edge] = uac_rectangle.length_alpha
    alpha[third_edge] = 4 * uac_rectangle.length_alpha * (0.75 - path.relative_lengths[third_edge])
    alpha[fourth_edge] = 0
    alpha += uac_rectangle.lower_left_corner[0]
    beta[first_edge] = 0
    beta[second_edge] = 4 * uac_rectangle.length_beta * (path.relative_lengths[second_edge] - 0.25)
    beta[third_edge] = uac_rectangle.length_beta
    beta[fourth_edge] = 4 * uac_rectangle.length_beta * (1 - path.relative_lengths[fourth_edge])
    beta += uac_rectangle.lower_left_corner[1]

    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path


# --------------------------------------------------------------------------------------------------
def compute_uacs_line(path: par.ParameterizedPath, uac_line: UACLine) -> UACPath:
    alpha = uac_line.start[0] + (uac_line.end[0] - uac_line.start[0]) * path.relative_lengths
    beta = uac_line.start[1] + (uac_line.end[1] - uac_line.start[1]) * path.relative_lengths
    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path
