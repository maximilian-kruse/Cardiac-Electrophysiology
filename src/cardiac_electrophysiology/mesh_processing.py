from dataclasses import dataclass
from numbers import Real

import fast_simplification as fs
import numpy as np
import pyvista as pv
import trimesh as tm


# ==================================================================================================
@dataclass
class FeatureTags:
    MV: int = None
    LAA: int = None
    LIPV: int = None
    LSPV: int = None
    RIPV: int = None
    RSPV: int = None


# ==================================================================================================
def convert_unstructured_to_polydata_mesh(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    point_data = mesh.point_data
    cell_data = mesh.cell_data
    polydata_mesh = pv.PolyData(mesh.points, mesh.cells)
    for key in point_data:
        polydata_mesh.point_data[key] = point_data[key]
    for key in cell_data:
        polydata_mesh.cell_data[key] = cell_data[key]

    return polydata_mesh


# --------------------------------------------------------------------------------------------------
def coarsen_mesh_with_feature_tags(
    mesh: pv.PolyData, decimation_factor: Real, num_smoothing_iters: int = 50
) -> pv.PolyData:
    mesh = mesh.cell_data_to_point_data()
    coarse_mesh = fs.simplify_mesh(mesh, target_reduction=decimation_factor)
    smoothened_mesh = coarse_mesh.smooth(n_iter=num_smoothing_iters, relaxation_factor=0.1)
    smoothened_mesh = smoothened_mesh.interpolate(mesh, strategy="closest_point")
    smoothened_mesh = smoothened_mesh.point_data_to_cell_data()
    smoothened_mesh = _make_feature_tags_discrete_after_interpolation(
        smoothened_mesh, extraction_threshold=0.5, lower_cutoff=0.1
    )
    smoothened_mesh = fix_feature_tag_defects(smoothened_mesh)
    return smoothened_mesh


# --------------------------------------------------------------------------------------------------
def fix_feature_tag_defects(mesh: pv.PolyData) -> pv.PolyData:
    mesh = _fix_feature_tag_dangles(mesh)
    mesh = _fix_feature_tag_spikes(mesh)
    mesh = _fix_body_tag_spikes(mesh)
    mesh = _fix_boundary_spikes(mesh)
    return mesh


# ==================================================================================================
def _make_feature_tags_discrete_after_interpolation(
    mesh: pv.PolyData, extraction_threshold: Real, lower_cutoff: Real = 0.1
) -> pv.PolyData:
    near_zero_tags = np.where(mesh.cell_data["anatomical_tags"] <= lower_cutoff)[0]
    mesh.cell_data["anatomical_tags"][near_zero_tags] = 0
    feature_meshes = mesh.extract_values(
        ranges=[lower_cutoff, float("inf")], scalars="anatomical_tags"
    )
    separated_meshes = feature_meshes.split_bodies()

    for feature_mesh in separated_meshes:
        fixed_feature_mesh = _extract_feature_tags_above_threshold(
            feature_mesh, extraction_threshold
        )
        mesh.cell_data["anatomical_tags"][feature_mesh.cell_data["vtkOriginalCellIds"]] = (
            fixed_feature_mesh.cell_data["anatomical_tags"]
        )
    return mesh


# --------------------------------------------------------------------------------------------------
def _extract_feature_tags_above_threshold(
    mesh: pv.PolyData, extraction_threshold: Real
) -> pv.PolyData:
    maximum_tag_value = np.max(mesh.cell_data["anatomical_tags"])
    threshold_value = extraction_threshold * maximum_tag_value
    feature_cell_inds = np.where(mesh.cell_data["anatomical_tags"] >= threshold_value)[0]
    body_cell_inds = np.setdiff1d(np.arange(mesh.number_of_cells), feature_cell_inds)
    mesh.cell_data["anatomical_tags"][feature_cell_inds] = maximum_tag_value
    mesh.cell_data["anatomical_tags"][body_cell_inds] = 0
    return mesh


# --------------------------------------------------------------------------------------------------
def _fix_boundary_spikes(mesh: pv.PolyData) -> None:
    num_neighbors_per_cell = _get_num_neighbors_per_cell(mesh)
    non_spike_cells = np.where(num_neighbors_per_cell > 1)[0]
    new_simplices = mesh.faces.reshape(-1, 4)[non_spike_cells]
    mesh.faces = new_simplices.flatten()
    mesh.remove_unused_points()

    return mesh


# --------------------------------------------------------------------------------------------------
def _fix_feature_tag_dangles(mesh: pv.PolyData) -> pv.PolyData:
    feature_tags = np.unique(mesh.cell_data["anatomical_tags"])
    feature_tags = feature_tags[feature_tags > 0]
    new_mesh = mesh.copy()
    new_mesh.cell_data["anatomical_tags"] = np.zeros(mesh.number_of_cells, dtype=int)

    for tag_value in feature_tags:
        feature_mesh = mesh.extract_values(tag_value, scalars="anatomical_tags")
        tm_mesh = tm.Trimesh(
            vertices=feature_mesh.points, faces=feature_mesh.cells.reshape(-1, 4)[:, 1:]
        )
        split_mesh = tm_mesh.split(only_watertight=False)
        num_cells = [submesh.faces.shape[0] for submesh in split_mesh]
        main_body = split_mesh[np.argmax(num_cells)]
        main_body.fill_holes()
        original_cell_inds = new_mesh.find_containing_cell(main_body.triangles_center)
        new_mesh.cell_data["anatomical_tags"][original_cell_inds] = tag_value

    return new_mesh


# --------------------------------------------------------------------------------------------------
def _fix_feature_tag_spikes(mesh: pv.PolyData) -> pv.PolyData:
    feature_tags = np.unique(mesh.cell_data["anatomical_tags"])
    feature_tags = feature_tags[feature_tags > 0]
    num_fixed_cells = 1

    while num_fixed_cells > 0:
        num_fixed_cells = 0
        for tag_value in feature_tags:
            feature_mesh = mesh.extract_values(tag_value, scalars="anatomical_tags")
            feature_mesh = convert_unstructured_to_polydata_mesh(feature_mesh)
            num_neighbors_per_cell = _get_num_neighbors_per_cell(feature_mesh)
            spike_cells = np.where(num_neighbors_per_cell == 1)[0]
            mesh.cell_data["anatomical_tags"][
                feature_mesh.cell_data["vtkOriginalCellIds"][spike_cells]
            ] = 0
            num_fixed_cells += spike_cells.shape[0]

    return mesh


# --------------------------------------------------------------------------------------------------
def _fix_body_tag_spikes(mesh: pv.PolyData) -> pv.PolyData:
    num_fixed_cells = 1

    while num_fixed_cells > 0:
        num_fixed_cells = 0
        body_mesh = mesh.extract_values(0, scalars="anatomical_tags")
        body_mesh = convert_unstructured_to_polydata_mesh(body_mesh)
        num_neighbors_per_cell = _get_num_neighbors_per_cell(body_mesh)
        spike_cells = np.where(num_neighbors_per_cell == 1)[0]
        original_spike_cells = body_mesh.cell_data["vtkOriginalCellIds"][spike_cells]
        new_tags = _get_different_neighbor_tag(mesh, original_spike_cells)
        mesh.cell_data["anatomical_tags"][original_spike_cells] = new_tags
        num_fixed_cells += spike_cells.shape[0]

    return mesh


# --------------------------------------------------------------------------------------------------
def _get_different_neighbor_tag(mesh: pv.PolyData, cell_inds: np.array) -> pv.PolyData:
    different_tags = []
    for ind in cell_inds:
        cell_tag = mesh.cell_data["anatomical_tags"][ind]
        cell_neighbors = mesh.cell_neighbors(ind)
        cell_neighbor_tags = mesh.cell_data["anatomical_tags"][cell_neighbors]
        different_tag = cell_neighbor_tags[cell_neighbor_tags != cell_tag][0]
        different_tags.append(different_tag)

    return np.array(different_tags)


# --------------------------------------------------------------------------------------------------
def _get_num_neighbors_per_cell(mesh: pv.PolyData) -> np.ndarray:
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    face_adjacency_data = tm_mesh.face_adjacency
    num_neighbors_per_cell = np.bincount(
        face_adjacency_data.flatten(), minlength=mesh.number_of_cells
    )
    return num_neighbors_per_cell
