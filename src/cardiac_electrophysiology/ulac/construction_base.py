from dataclasses import dataclass

import igraph as ig
import numpy as np
import pyvista as pv
import trimesh as tm
from numba import njit
from numba.typed import List


# ==================================================================================================
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


@dataclass
class Submesh:
    inds: np.ndarray = None
    connectivity: np.ndarray = None


@dataclass
class UACSubmesh(Submesh):
    alpha: np.ndarray = None
    beta: np.ndarray = None


# ==================================================================================================
def get_feature_boundary(
    mesh: pv.PolyData, feature_tag: int, coincides_with_geometry_boundary: bool
) -> tuple[np.ndarray, np.ndarray]:
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

    ordered_boundary_inds = _construct_ordered_path_from_indices(mesh, feature_boundary_inds)
    return ordered_boundary_inds


# --------------------------------------------------------------------------------------------------
def _construct_ordered_path_from_indices(mesh: pv.PolyData, path_indices: np.ndarray) -> np.ndarray:
    tm_mesh = tm.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    path_edges = tm_mesh.edges[np.isin(tm_mesh.edges, path_indices).all(axis=1)].flatten()
    local_edges = np.array([np.where(path_indices == ind)[0][0] for ind in path_edges])
    local_edges = local_edges.reshape(-1, 2)
    graph = ig.Graph(local_edges, directed=False)
    ordered_path, _ = graph.dfs(0)
    return path_indices[ordered_path]


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData,
    subset_one: np.ndarray,
    subset_two: np.ndarray,
    inadmissible_contact_set: np.ndarray,
    inadmissible_along_set: np.ndarray,
):
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    edges = tm_mesh.edges_unique
    edge_lengths = tm_mesh.edges_unique_length
    inadmissible_contact_edges = np.where(np.isin(edges, inadmissible_contact_set).any(axis=1))[0]
    inadmissible_along_edges = np.where(np.isin(edges, inadmissible_along_set).all(axis=1))[0]
    inadmissible_edges = np.unique(
        np.concatenate([inadmissible_contact_edges, inadmissible_along_edges])
    )
    admissible_edges = np.delete(edges, inadmissible_edges, axis=0)
    admissible_edges_lengths = np.delete(edge_lengths, inadmissible_edges, axis=0)

    graph = ig.Graph(admissible_edges, directed=False)
    distances = np.array(graph.distances(subset_one, subset_two, weights=admissible_edges_lengths))
    rel_start_point, rel_end_point = np.unravel_index(np.argmin(distances), distances.shape)
    graph = ig.Graph(admissible_edges, directed=False)
    shortest_path = np.array(
        graph.get_shortest_paths(
            subset_one[rel_start_point],
            to=subset_two[rel_end_point],
            weights=admissible_edges_lengths,
        )[0]
    )
    return np.array(shortest_path, dtype=int)


# --------------------------------------------------------------------------------------------------
def extract_region_from_boundary(
    mesh: pv.PolyData, boundary_inds: np.ndarray, outside_inds: np.ndarray
) -> np.ndarray:
    mesh_points_without_boundary = np.setdiff1d(np.arange(mesh.number_of_points), boundary_inds)
    mesh_without_boundary = mesh.extract_points(mesh_points_without_boundary, adjacent_cells=False)
    submeshes = mesh_without_boundary.split_bodies()
    coinciding_outside_inds = np.where(
        submeshes[0].point_data["vtkOriginalPointIds"] == outside_inds[0]
    )[0]
    inside_mesh = submeshes[0] if coinciding_outside_inds.size == 0 else submeshes[1]
    seed_ind = inside_mesh.point_data["vtkOriginalPointIds"][0]

    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4])
    boundary_ind_mask = np.zeros(mesh.number_of_points, dtype=np.bool_)
    boundary_ind_mask[boundary_inds] = True

    graph = ig.Graph(tm_mesh.edges_unique, directed=False)
    adjacency = graph.get_adjacency_sparse()
    adjacency = adjacency.tocoo()
    _, ind_starts = np.unique(adjacency.row, return_index=True)
    ind_neighbors = adjacency.col
    submesh_inds = _extract_region_from_boundary(
        ind_starts, ind_neighbors, seed_ind, boundary_ind_mask, mesh.number_of_points
    )
    pv_submesh = mesh.extract_points(submesh_inds, adjacent_cells=False)
    connectivity = np.array(pv_submesh.cells.reshape(-1, 4)[:, 1:])
    submesh = Submesh(inds=submesh_inds, connectivity=connectivity)

    return submesh


# --------------------------------------------------------------------------------------------------
@njit
def _extract_region_from_boundary(
    index_starts: np.ndarray,
    index_neighbors: np.ndarray,
    seed_ind: int,
    boundary_ind_mask: np.ndarray,
    number_of_points: int,
) -> np.ndarray:
    visited_inds = np.zeros(number_of_points, dtype=np.bool_)
    active_list = List([seed_ind])
    neighbor_inds = List([seed_ind])
    neighbor_inds.clear()

    while active_list:
        current_ind = active_list.pop()
        visited_inds[current_ind] = True
        # Get all admissible neighbor edges
        start_ind = index_starts[current_ind]
        end_ind = (
            index_starts[current_ind + 1]
            if current_ind + 1 < index_starts.size
            else index_neighbors.size
        )
        is_boundary = boundary_ind_mask[current_ind]
        for i in range(start_ind, end_ind):
            if is_boundary and (not boundary_ind_mask[index_neighbors[i]]):
                continue
            neighbor_inds.append(index_neighbors[i])
        # Add unvisited neighbors to active list
        for i in range(len(neighbor_inds)):
            neighbor_ind = neighbor_inds[i]
            if not visited_inds[neighbor_ind]:
                active_list.append(neighbor_ind)
        neighbor_inds.clear()

    return np.where(visited_inds)[0]


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
    cumulative_lengths = np.cumsum(edge_lengths)
    relative_path_length = np.zeros(coordinates.shape[0])
    relative_path_length[1:] = cumulative_lengths / cumulative_lengths[-1]
    segmented_lengths = np.zeros(coordinates.shape[0])

    splitting_inds = relative_marker_inds[1:]
    if 1.0 in marker_values:
        splitting_inds = splitting_inds[:-1]
        segment_boundaries = marker_values
    else:
        closing_edge_length = np.linalg.norm(coordinates[0] - coordinates[-1])
        end_marker = relative_path_length[-1] / (
            relative_path_length[-1] + closing_edge_length / cumulative_lengths[-1]
        )
        segment_boundaries = [*marker_values, end_marker]
    segments = np.split(np.arange(path.size), splitting_inds)

    for i, segment_inds in enumerate(segments):
        include_endpoint = i == len(segments) - 1
        start_value = segment_boundaries[i]
        end_value = segment_boundaries[i + 1]
        num_points = segment_inds.size
        segment_parameterization = np.linspace(
            start_value, end_value, num_points, endpoint=include_endpoint
        )
        segmented_lengths[segment_inds] = segment_parameterization

    parameterized_path = ParameterizedPath(path, segmented_lengths)
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
    uac_segment_boundaries = np.array(segment_uacs)

    if 1.0 not in segment_points:
        end_uacs = uac_segment_boundaries[-1] + path.relative_lengths[-1] * (
            uac_segment_boundaries[0] - uac_segment_boundaries[-1]
        )
        uac_segment_boundaries = np.vstack((uac_segment_boundaries, end_uacs))

    alpha, beta = [], []
    for i, segment in enumerate(segments):
        next_segment = segments[i + 1] if i < len(segments) - 1 else None
        start_point = uac_segment_boundaries[i]
        if next_segment is not None:
            end_point = uac_segment_boundaries[i] + segment[-1] / next_segment[0] * (
                uac_segment_boundaries[i + 1] - uac_segment_boundaries[i]
            )
        else:
            end_point = uac_segment_boundaries[i + 1]
        segment_alpha, segment_beta = _compute_uacs_edge(segment, start_point, end_point)
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
