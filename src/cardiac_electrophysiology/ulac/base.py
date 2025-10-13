from dataclasses import dataclass

import igl
import networkx as nx
import numpy as np
import pyvista as pv
import trimesh as tm


# ==================================================================================================
@dataclass
class Submesh:
    inds: np.ndarray = None
    connectivity: np.ndarray = None


@dataclass
class Marker:
    ind: int = None
    uacs: tuple[float, float] = None


@dataclass
class ParameterizedPath:
    inds: np.ndarray = None
    relative_lengths: np.ndarray = None


@dataclass
class UACPath(ParameterizedPath):
    alpha: np.ndarray = None
    beta: np.ndarray = None


# ==================================================================================================
def get_feature_boundary(
    mesh: pv.PolyData, feature_tag: int, coincides_with_geometry_boundary: bool
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

    if coincides_with_geometry_boundary:
        feature_boundary_inds = np.intersect1d(feature_boundary_inds, geometry_boundary_inds)
    else:
        feature_boundary_inds = np.setdiff1d(feature_boundary_inds, geometry_boundary_inds)

    ordered_boundary_inds = _construct_ordered_path_from_indices(tm_mesh, feature_boundary_inds)
    return ordered_boundary_inds


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


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData, subset_one: np.ndarray, subset_two: np.ndarray, inadmissible_set: np.ndarray
):
    vertices = np.array(mesh.points)
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:])
    distances_one = igl.exact_geodesic(V=vertices, F=simplices, VS=subset_one, VT=subset_two)
    end_point = subset_two[np.argmin(distances_one)]
    distances_two = igl.exact_geodesic(
        V=vertices, F=simplices, VS=np.array([end_point]), VT=subset_one
    )
    start_point = subset_one[np.argmin(distances_two)]
    inadmissible_points = np.setdiff1d(inadmissible_set, np.array([start_point, end_point]))
    tm_mesh = tm.Trimesh(vertices, simplices)
    edges = tm_mesh.edges_unique
    edge_lengths = tm_mesh.edges_unique_length
    admissible_edges = edges[~np.isin(edges, inadmissible_points).any(axis=1)]
    admissible_edges_lengths = edge_lengths[~np.isin(edges, inadmissible_points).any(axis=1)]
    weighted_edges = np.hstack((admissible_edges, admissible_edges_lengths[:, None]))
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges, "length")
    optimal_path = nx.shortest_path(graph, source=start_point, target=end_point, weight="length")
    return np.array(optimal_path, dtype=int)


# --------------------------------------------------------------------------------------------------
def split_from_enclosing_boundary(mesh: pv.PolyData, boundary_inds: np.ndarray):
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:])
    not_boundary_inds = np.setdiff1d(np.arange(mesh.number_of_points), boundary_inds)
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    mesh_edges = tm_mesh.edges_unique
    connected_components = tm.graph.connected_components(mesh_edges, nodes=not_boundary_inds)
    submeshes_with_boundary = []

    for component_inds in connected_components:
        point_inds_with_boundary = np.sort(np.hstack([component_inds, boundary_inds]).astype(int))
        contained_cell_inds = np.where(np.isin(simplices, point_inds_with_boundary).all(axis=1))[0]
        submesh_with_boundary = mesh.extract_cells(contained_cell_inds)
        original_point_inds = submesh_with_boundary.point_data["vtkOriginalPointIds"]
        connectivity = np.array(submesh_with_boundary.cells.reshape(-1, 4)[:, 1:])
        submeshes_with_boundary.append(Submesh(original_point_inds, connectivity))

    return submeshes_with_boundary


# ==================================================================================================
def parameterize_path(
    mesh: pv.PolyData, path: np.ndarray, marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    reordered_path, relative_marker_inds = _reorder_path_by_markers(
        path, marker_inds, marker_values
    )
    parameterized_path = _parameterize_by_relative_length(
        mesh, reordered_path, relative_marker_inds, marker_values
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def _reorder_path_by_markers(
    path: np.ndarray,
    marker_inds: list[int],
    marker_values: list[float],
) -> tuple[np.ndarray, list[int]]:
    relative_start_ind_location = np.where(path == marker_inds[0])[0]
    ordered_path = np.roll(path, -relative_start_ind_location)

    relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        ordered_path = np.roll(np.flip(ordered_path), 1)
        relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    return ordered_path, relative_marker_inds


# --------------------------------------------------------------------------------------------------
def _parameterize_by_relative_length(
    mesh: pv.PolyData, path: np.ndarray, relative_marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    coordinates = np.array(mesh.points[path])
    edge_lengths = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    relative_path_length = np.zeros(coordinates.shape[0])
    relative_path_length[1:] = np.cumsum(edge_lengths)
    relative_path_length /= relative_path_length[-1]
    relative_lengths = np.zeros(coordinates.shape[0])

    splitting_inds = relative_marker_inds[1:]
    if 1.0 in marker_values:
        splitting_inds = splitting_inds[:-1]
    segments = np.split(np.arange(path.size), splitting_inds)
    segment_boundaries = [*marker_values, 1.0] if 1.0 not in marker_values else marker_values

    for i, segment_inds in enumerate(segments):
        include_endpoint = i == len(segments) - 1
        start_value = segment_boundaries[i]
        end_value = segment_boundaries[i + 1]
        num_points = segment_inds.size
        segment_parameterization = np.linspace(
            start_value, end_value, num_points, endpoint=include_endpoint
        )
        relative_lengths[segment_inds] = segment_parameterization

    parameterized_path = ParameterizedPath(path, relative_lengths)
    return parameterized_path


# ==================================================================================================
def compute_uacs_polygon(
    path: ParameterizedPath,
    segment_points: list[float],
    segment_uacs: list[tuple[float, float]],
) -> UACPath:
    splitting_points = segment_points[1:]
    if 1.0 in segment_points:
        splitting_points = splitting_points[:-1]
    segment_indices = np.searchsorted(path.relative_lengths, splitting_points)
    segments = np.split(path.relative_lengths, segment_indices)
    if 1.0 not in segment_points:
        segment_uacs.append(segment_uacs[0])

    alpha, beta = [], []
    for i, segment in enumerate(segments):
        segment_alpha, segment_beta = _compute_uacs_edge(
            segment, segment_uacs[i], segment_uacs[i + 1]
        )
        alpha.append(segment_alpha)
        beta.append(segment_beta)
    alpha = np.concatenate(alpha)
    beta = np.concatenate(beta)

    uac_path = UACPath(
        inds=path.inds, relative_lengths=path.relative_lengths, alpha=alpha, beta=beta
    )
    return uac_path


# --------------------------------------------------------------------------------------------------
def compute_uacs_line(path: ParameterizedPath, start_point: tuple, end_point: tuple) -> UACPath:
    alpha, beta = _compute_uacs_edge(path.relative_lengths, start_point, end_point)
    uac_path = UACPath(
        inds=path.inds, relative_lengths=path.relative_lengths, alpha=alpha, beta=beta
    )
    return uac_path


# --------------------------------------------------------------------------------------------------
def _compute_uacs_edge(
    relative_length: np.ndarray, start_point: tuple, end_point: tuple
) -> tuple[np.ndarray, np.ndarray]:
    length_range = np.max(relative_length) - np.min(relative_length)
    scaled_length = (relative_length - np.min(relative_length)) / length_range
    alpha = start_point[0] + (end_point[0] - start_point[0]) * scaled_length
    beta = start_point[1] + (end_point[1] - start_point[1]) * scaled_length
    return alpha, beta
