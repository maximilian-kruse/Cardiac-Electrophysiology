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


# --------------------------------------------------------------------------------------------------
def split_from_enclosing_boundary(mesh: pv.PolyData, boundary_inds: np.ndarray):
    simplices = convert_pv_cells_to_numpy_cells(mesh.faces)
    not_boundary_inds = np.setdiff1d(np.arange(mesh.number_of_points), boundary_inds)
    not_boundary_submesh = mesh.extract_points(not_boundary_inds, adjacent_cells=False)
    boundary_submesh_with_padding = mesh.extract_points(boundary_inds, adjacent_cells=True)
    split_mesh = not_boundary_submesh.split_bodies()
    submeshes_with_boundary = []

    for submesh in split_mesh:
        interior_cell_inds = submesh.cell_data["vtkOriginalCellIds"]
        padded_boundary_cell_inds = boundary_submesh_with_padding.cell_data["vtkOriginalCellIds"]
        contained_cell_inds = np.unique(
            np.concatenate([interior_cell_inds, padded_boundary_cell_inds])
        )
        submesh_inflated = mesh.extract_cells(contained_cell_inds)
        submesh_outer_boundary = submesh_inflated.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        outer_boundary_inds = submesh_inflated.point_data["vtkOriginalPointIds"][
            submesh_outer_boundary.point_data["vtkOriginalPointIds"]
        ]
        cells_touching_outer_boundary = np.where(
            np.isin(simplices, outer_boundary_inds).any(axis=1)
        )[0]
        cells_touching_boundary = np.where(np.isin(simplices, boundary_inds).any(axis=1))[0]
        cells_touching_both = np.intersect1d(cells_touching_outer_boundary, cells_touching_boundary)
        cells_to_keep = np.setdiff1d(contained_cell_inds, cells_touching_both)
        submesh_with_boundary = mesh.extract_cells(cells_to_keep)
        submeshes_with_boundary.append(submesh_with_boundary)

    return submeshes_with_boundary


# --------------------------------------------------------------------------------------------------
def get_local_point_inds(submesh: pv.PolyData, global_point_inds: np.ndarray) -> np.ndarray:
    original_inds = submesh.point_data["vtkOriginalPointIds"]
    local_point_inds = np.array(
        [
            np.where(original_inds == ind)[0][0]
            for ind in global_point_inds
            if ind in original_inds
        ]
    )
    return local_point_inds


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
