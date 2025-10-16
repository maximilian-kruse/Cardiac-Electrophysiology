from dataclasses import dataclass

import igraph as ig
import numpy as np
import pyvista as pv
import trimesh as tm


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
    return ordered_boundary_inds[:-1]


# --------------------------------------------------------------------------------------------------
def _construct_ordered_path_from_indices(
    mesh: pv.PolyData, path_indices: np.ndarray
) -> np.ndarray:
    tm_mesh = tm.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    path_edges = tm_mesh.edges[np.isin(tm_mesh.edges, path_indices).all(axis=1)].flatten()
    local_edges = np.array([np.where(path_indices == ind)[0][0] for ind in path_edges])
    local_edges = local_edges.reshape(-1, 2)
    graph = ig.Graph(local_edges, directed=False)
    ordered_path, _ = graph.dfs(0)
    return path_indices[ordered_path]


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData, subset_one: np.ndarray, subset_two: np.ndarray, inadmissible_set: np.ndarray
):
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    edges = tm_mesh.edges_unique
    edge_lengths = tm_mesh.edges_unique_length

    graph = ig.Graph(edges, directed=False)
    distances = np.array(graph.distances(subset_one, subset_two, weights=edge_lengths))
    rel_start_point, rel_end_point = np.unravel_index(np.argmin(distances), distances.shape)
    start_point = subset_one[rel_start_point]
    end_point = subset_two[rel_end_point]

    inadmissible_points = np.setdiff1d(inadmissible_set, np.array([start_point, end_point]))
    admissible_edges = edges[~np.isin(edges, inadmissible_points).any(axis=1)]
    admissible_edges_lengths = edge_lengths[~np.isin(edges, inadmissible_points).any(axis=1)]
    graph = ig.Graph(admissible_edges, directed=False)
    shortest_path = np.array(graph.get_shortest_paths(
        subset_one[rel_start_point], to=subset_two[rel_end_point], weights=admissible_edges_lengths
    )[0])
    return np.array(shortest_path, dtype=int)


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
