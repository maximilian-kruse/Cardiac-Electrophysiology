import operator
from collections.abc import Callable
from functools import reduce

import igl
import numpy as np
import pyvista as pv

from .deprecated import configuration as config
from .. import construction_base as base


# ==================================================================================================
def apply_to_dict(func: Callable, input_arg: object) -> Callable:
    if isinstance(input_arg, dict):
        return {key: apply_to_dict(func, value) for key, value in input_arg.items()}
    return func(input_arg)


# --------------------------------------------------------------------------------------------------
def construct_segmentation(mesh: pv.PolyData, feature_tags: dict[str, int]) -> dict:
    boundary_paths = extract_boundary_paths(mesh, feature_tags)
    direct_connection_paths = construct_connection_paths(
        mesh, config.direct_connection_path_configs, boundary_paths
    )
    indirect_connection_paths = construct_connection_paths(
        mesh, config.indirect_connection_path_configs, {**boundary_paths, **direct_connection_paths}
    )
    segmentation_paths = {**boundary_paths, **direct_connection_paths, **indirect_connection_paths}
    return segmentation_paths


# --------------------------------------------------------------------------------------------------
def extract_boundary_paths(
    mesh: pv.PolyData,
    feature_tags: dict["str", int],
) -> np.ndarray:
    def _extract_boundary_paths(boundary_path_config: config.BoundaryPathConfig) -> dict:
        tag_value = feature_tags[boundary_path_config.feature_tag]
        is_mesh_boundary = boundary_path_config.coincides_with_mesh_boundary
        boundary_path = base.get_feature_boundary(mesh, tag_value, is_mesh_boundary)
        return boundary_path

    boundary_paths = apply_to_dict(_extract_boundary_paths, config.boundary_path_configs)
    return boundary_paths


# --------------------------------------------------------------------------------------------------
def construct_connection_paths(
    mesh: pv.PolyData,
    path_configs: dict,
    computed_paths: dict,
) -> dict:
    def _construct_connection_path(path_config: config.ConnectionPathConfig) -> np.ndarray:
        boundary_sets = []
        for boundary_set in (path_config.start, path_config.end):
            if isinstance(boundary_set, tuple):
                bs = reduce(operator.getitem, boundary_set, computed_paths)
            elif isinstance(boundary_set, config.PointConfig):
                boundary_path = reduce(
                    operator.getitem, boundary_set.containing_path, computed_paths
                )
                bs = np.array((boundary_path[boundary_set.index],), dtype=int)
            boundary_sets.append(bs)
        inadmissible_sets = [
            reduce(operator.getitem, inadmissible, computed_paths)
            for inadmissible in path_config.inadmissible
        ]
        inadmissible_set = np.unique(np.concatenate(inadmissible_sets))
        path = base.construct_shortest_path_between_subsets(mesh, *boundary_sets, inadmissible_set)
        return path

    connection_paths = apply_to_dict(_construct_connection_path, path_configs)
    return connection_paths


# --------------------------------------------------------------------------------------------------
def get_markers(segmentation_paths: dict) -> dict:
    def _get_marker(marker_config: config.MarkerConfig) -> base.Marker:
        path = reduce(operator.getitem, marker_config.point.containing_path, segmentation_paths)
        marker_index = path[marker_config.point.index]
        marker = base.Marker(ind=marker_index, uacs=marker_config.uacs)
        return marker

    markers = apply_to_dict(_get_marker, config.marker_configs)
    return markers


# --------------------------------------------------------------------------------------------------
def parameterize_paths(
    mesh: pv.PolyData,
    segmentation_paths: dict,
    markers: dict,
) -> dict:
    def _parameterize_path(path_config: config.ParameterizationConfig) -> base.ParameterizedPath:
        path = reduce(operator.getitem, path_config.path, segmentation_paths)
        path_markers = [reduce(operator.getitem, config, markers) for config in path_config.markers]
        marker_inds = [marker.ind for marker in path_markers]
        parameterized_path = base.parameterize_path(
            mesh, path, marker_inds, path_config.marker_values
        )
        return parameterized_path

    parameterized_paths = apply_to_dict(_parameterize_path, config.parameterization_configs)
    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def construct_uac_paths(parameterized_paths: dict, markers: dict) -> dict:
    def _construct_uac_path(path_config: config.ParameterizationConfig) -> base.UACPath:
        path = reduce(operator.getitem, path_config.path, parameterized_paths)
        path_markers = [reduce(operator.getitem, config, markers) for config in path_config.markers]
        marker_uacs = [marker.uacs for marker in path_markers]
        marker_values = path_config.marker_values
        if len(path_markers) == 2:
            start_uac, end_uac = marker_uacs
            uac_path = base.compute_uacs_line(path, start_uac, end_uac)
        else:
            uac_path = base.compute_uacs_polygon(path, marker_values, marker_uacs)
        return uac_path

    uac_paths = apply_to_dict(_construct_uac_path, config.parameterization_configs)
    return uac_paths


# --------------------------------------------------------------------------------------------------
def get_patch_boundaries(uac_paths: dict) -> dict:
    def _get_patch_boundary(boundary_config: config.PatchBoundaryConfig) -> base.UACPath:
        boundary_paths = [
            reduce(operator.getitem, path, uac_paths) for path in boundary_config.paths
        ]
        indices = []
        alpha = []
        beta = []
        for path, portion in zip(boundary_paths, boundary_config.portions, strict=True):
            relevant_section = np.where(
                (path.relative_lengths >= portion[0]) & (path.relative_lengths <= portion[1])
            )[0]
            indices.append(path.inds[relevant_section])
            alpha.append(path.alpha[relevant_section])
            beta.append(path.beta[relevant_section])

        unique_inds, unique_mask = np.unique(np.concatenate(indices), return_index=True)
        unique_alpha = np.concatenate(alpha)[unique_mask]
        unique_beta = np.concatenate(beta)[unique_mask]
        patch_boundary = base.UACPath(inds=unique_inds, alpha=unique_alpha, beta=unique_beta)
        return patch_boundary

    uac_patch_boundaries = apply_to_dict(_get_patch_boundary, config.patch_boundary_configs)
    return uac_patch_boundaries


# --------------------------------------------------------------------------------------------------
def compute_patch_uacs(mesh: pv.PolyData, patch_boundaries: dict, segmentation_paths: dict) -> dict:
    def _compute_patch_uac(patch_config: config.PatchConfig) -> tuple[float, float]:
        boundary_path = reduce(operator.getitem, patch_config.boundary, patch_boundaries)
        submeshes = base.split_from_enclosing_boundary(mesh, boundary_path.inds)
        outside_path = reduce(operator.getitem, patch_config.outside, segmentation_paths)
        if np.isin(submeshes[0].inds, outside_path).any():
            inside_submesh = submeshes[1]
        else:
            inside_submesh = submeshes[0]

        vertices = np.array(mesh.points[inside_submesh.inds])
        simplices = inside_submesh.connectivity
        boundary_inds = np.array(
            [np.where(inside_submesh.inds == ind)[0][0] for ind in boundary_path.inds]
        )
        uac_coordinates = np.hstack((boundary_path.alpha[:, None], boundary_path.beta[:, None]))
        harmonic_map = igl.harmonic(
            V=vertices, F=simplices, b=boundary_inds, bc=uac_coordinates, k=1,
        )
        uac_submesh = base.UACSubmesh(
            inds=inside_submesh.inds,
            connectivity=simplices,
            alpha=harmonic_map[:, 0],
            beta=harmonic_map[:, 1],
        )
        return uac_submesh

    patch_uacs = apply_to_dict(_compute_patch_uac, config.patch_configs)
    return patch_uacs
