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
