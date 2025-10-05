from numbers import Real

import fast_simplification as fs
import numpy as np
import pyvista as pv
import trimesh as tm


# ==================================================================================================
def convert_unstructured_to_polydata_mesh(input_mesh: pv.UnstructuredGrid) -> pv.PolyData:
    point_data = input_mesh.point_data
    cell_data = input_mesh.cell_data
    polydata_mesh = pv.PolyData(input_mesh.points, input_mesh.cells)
    for key in point_data:
        polydata_mesh.point_data[key] = point_data[key]
    for key in cell_data:
        polydata_mesh.cell_data[key] = cell_data[key]

    return polydata_mesh


# --------------------------------------------------------------------------------------------------
def convert_numpy_cells_to_pv_cells(
    numpy_cells: np.ndarray,
) -> np.ndarray:
    num_vertices_array = 3 * np.ones(numpy_cells.shape[0])
    pv_cell_array = np.hstack((num_vertices_array[:, np.newaxis], numpy_cells))
    pv_cell_array = pv_cell_array.flatten().astype(int)

    return pv_cell_array


# --------------------------------------------------------------------------------------------------
def convert_pv_cells_to_numpy_cells(
    pv_cells: np.ndarray,
) -> np.ndarray:
    numpy_cells = pv_cells.reshape(-1, 4)[:, 1:]

    return numpy_cells


# ==================================================================================================
def coarsen_mesh(
    input_mesh: pv.PolyData, decimation_factor: Real, num_smoothing_iters: int = 50
) -> pv.PolyData:
    input_mesh = input_mesh.cell_data_to_point_data()
    coarse_mesh = fs.simplify_mesh(input_mesh, target_reduction=decimation_factor)
    smoothened_mesh = coarse_mesh.smooth(n_iter=num_smoothing_iters, relaxation_factor=0.1)
    smoothened_mesh = smoothened_mesh.interpolate(input_mesh, strategy="closest_point")
    smoothened_mesh = smoothened_mesh.point_data_to_cell_data()
    smoothened_mesh = _fix_feature_tags_after_interpolation(
        smoothened_mesh, extraction_threshold=0.5
    )
    smoothened_mesh = _remove_feature_boundary_spikes(smoothened_mesh)

    return smoothened_mesh


# ==================================================================================================
def _fix_feature_tags_after_interpolation(
    input_mesh: pv.PolyData, extraction_threshold: Real, lower_cutoff: Real = 0.1
) -> pv.PolyData:
    near_zero_tags = np.where(input_mesh.cell_data["anatomical_tags"] <= lower_cutoff)[0]
    input_mesh.cell_data["anatomical_tags"][near_zero_tags] = 0
    separated_meshes = _get_feature_submeshes(input_mesh, lower_cutoff)

    for feature_mesh in separated_meshes:
        fixed_feature_mesh = _extract_feature_tags_above_threshold(
            feature_mesh, extraction_threshold
        )
        input_mesh.cell_data["anatomical_tags"][feature_mesh.cell_data["vtkOriginalCellIds"]] = (
            fixed_feature_mesh.cell_data["anatomical_tags"]
        )

    return input_mesh


# --------------------------------------------------------------------------------------------------
def _remove_feature_boundary_spikes(input_mesh: pv.PolyData) -> pv.PolyData:
    sorted_cell_adjacency_data, cell_inds, num_neighbors = _get_cell_adjacencies(input_mesh)
    non_boundary_cells = cell_inds[num_neighbors == 3]
    non_boundary_adjacency_data = sorted_cell_adjacency_data[
        np.isin(sorted_cell_adjacency_data[:, 0], non_boundary_cells)
    ]
    neighbors_per_cell = non_boundary_adjacency_data[:, 1].reshape(-1, 3)

    # Remove body spikes
    cell_tags = input_mesh.cell_data["anatomical_tags"][non_boundary_cells]
    neighbor_tags = input_mesh.cell_data["anatomical_tags"][neighbors_per_cell]
    coinciding_neighbor_mask = neighbor_tags == cell_tags[:, np.newaxis]
    num_coinciding_neighbors = np.sum(coinciding_neighbor_mask, axis=1)
    spike_inds = np.where(num_coinciding_neighbors == 1)[0]
    body_spike_inds = spike_inds[cell_tags[spike_inds] == 0]
    first_nonconforming_neighbor_ind = np.argmin(coinciding_neighbor_mask[body_spike_inds], axis=1)
    first_non_conforming_neighbor_tag = neighbor_tags[
        body_spike_inds, first_nonconforming_neighbor_ind
    ]
    input_mesh.cell_data["anatomical_tags"][non_boundary_cells[body_spike_inds]] = (
        first_non_conforming_neighbor_tag
    )

    # Remove feature spikes
    cell_tags = input_mesh.cell_data["anatomical_tags"][non_boundary_cells]
    neighbor_tags = input_mesh.cell_data["anatomical_tags"][neighbors_per_cell]
    coinciding_neighbor_mask = neighbor_tags == cell_tags[:, np.newaxis]
    num_coinciding_neighbors = np.sum(coinciding_neighbor_mask, axis=1)
    spike_inds = np.where(num_coinciding_neighbors == 1)[0]
    feature_spike_inds = non_boundary_cells[spike_inds[cell_tags[spike_inds] != 0]]
    input_mesh.cell_data["anatomical_tags"][feature_spike_inds] = 0

    return input_mesh


# --------------------------------------------------------------------------------------------------
def _get_feature_submeshes(input_mesh: pv.PolyData, lower_cutoff: Real = 0.1) -> pv.MultiBlock:
    feature_meshes = input_mesh.extract_values(
        ranges=[lower_cutoff, float("inf")], scalars="anatomical_tags"
    )
    separated_meshes = feature_meshes.split_bodies()

    return separated_meshes


# --------------------------------------------------------------------------------------------------
def _extract_feature_tags_above_threshold(
    input_feature_mesh: pv.PolyData, extraction_threshold: Real
) -> pv.PolyData:
    maximum_tag_value = np.max(input_feature_mesh.cell_data["anatomical_tags"])
    threshold_value = extraction_threshold * maximum_tag_value
    feature_cell_inds = np.where(
        input_feature_mesh.cell_data["anatomical_tags"] >= threshold_value
    )[0]
    body_cell_inds = np.setdiff1d(np.arange(input_feature_mesh.number_of_cells), feature_cell_inds)
    input_feature_mesh.cell_data["anatomical_tags"][feature_cell_inds] = maximum_tag_value
    input_feature_mesh.cell_data["anatomical_tags"][body_cell_inds] = 0

    return input_feature_mesh


# --------------------------------------------------------------------------------------------------
def _get_cell_adjacencies(input_mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tm_mesh = tm.Trimesh(
        vertices=input_mesh.points, faces=convert_pv_cells_to_numpy_cells(input_mesh.faces)
    )
    cell_adjacency_data_unidirectional = tm_mesh.face_adjacency
    cell_adjacency_data_reversed = cell_adjacency_data_unidirectional[:, ::-1]
    cell_adjacency_data = np.vstack(
        (cell_adjacency_data_unidirectional, cell_adjacency_data_reversed)
    )
    sorting_mask = np.argsort(cell_adjacency_data[:, 0])
    sorted_cell_adjacency_data = cell_adjacency_data[sorting_mask]
    cell_inds, num_neighbors = np.unique(sorted_cell_adjacency_data[:, 0], return_counts=True)

    return sorted_cell_adjacency_data, cell_inds, num_neighbors
