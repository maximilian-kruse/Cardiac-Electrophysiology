import copy
import operator
from collections.abc import Callable
from functools import partial, reduce

import numpy as np
import pyvista as pv

from . import base
from . import configuration as config


# ==================================================================================================
def apply_to_dict(func: Callable, input_arg: object) -> Callable:
    if isinstance(input_arg, dict):
        return {key: apply_to_dict(func, value) for key, value in input_arg.items()}
    return func(input_arg)


# ==================================================================================================
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
    def _construct_connection_path(path_config: config.ConnectionPathConfig) -> dict:
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
        path = base.construct_shortest_path_between_subsets(mesh, *boundary_sets)
        return path

    connection_paths = apply_to_dict(_construct_connection_path, path_configs)
    return connection_paths


# # ==================================================================================================
# def parameterize_pv_boundary_paths(
#     mesh: pv.PolyData,
#     pv_boundary_paths: data.PVBoundaryPaths,
#     pv_segment_markers: data.PVSegmentMarkers,
# ) -> data.ParameterizedPVBoundaryPaths:
#     parameterized_pv_boundary_paths = data.ParameterizedPVBoundaryPaths()
#     parameterization_specs = spec.parameterization_specs["PV"]
#     start_ind_str = parameterization_specs["start"]
#     marker_ind_str = parameterization_specs["markers"]
#     marker_values = parameterization_specs["marker_values"]

#     for pv_boundary_field in fields(parameterized_pv_boundary_paths):
#         path = getattr(pv_boundary_paths, pv_boundary_field.name)
#         path_markers = getattr(pv_segment_markers, pv_boundary_field.name)

#         ordered_path, relative_marker_inds = base.reorder_path_by_markers(
#             path,
#             start_ind=getattr(path_markers, start_ind_str),
#             marker_inds=[getattr(path_markers, m) for m in marker_ind_str],
#             marker_values=marker_values,
#         )
#         parameterized_path = base.parameterize_path_by_relative_length(
#             mesh,
#             ordered_path,
#             relative_marker_inds=relative_marker_inds,
#             marker_values=marker_values,
#         )
#         setattr(parameterized_pv_boundary_paths, pv_boundary_field.name, parameterized_path)

#     return parameterized_pv_boundary_paths


# # --------------------------------------------------------------------------------------------------
# def parameterize_laa_boundary_path(
#     mesh: pv.PolyData, laa_boundary_path: np.ndarray, laa_markers: data.LAAMarkers
# ) -> base.ParameterizedPath:
#     parameterization_specs = spec.parameterization_specs["LAA"]
#     start_ind_str = parameterization_specs["start"]
#     marker_ind_str = parameterization_specs["markers"]
#     marker_values = parameterization_specs["marker_values"]

#     start_ind = getattr(laa_markers, start_ind_str)
#     marker_inds = [getattr(laa_markers, m) for m in marker_ind_str]
#     reordered_path, relative_marker_inds = base.reorder_path_by_markers(
#         laa_boundary_path,
#         start_ind=start_ind,
#         marker_inds=marker_inds,
#         marker_values=marker_values,
#     )
#     parameterized_path = base.parameterize_path_by_relative_length(
#         mesh,
#         reordered_path,
#         relative_marker_inds=[relative_marker_inds[0]],
#         marker_values=[marker_values[0]],
#     )
#     return parameterized_path


# # --------------------------------------------------------------------------------------------------
# def parameterize_mv_boundary_path(
#     mesh: pv.PolyData, mv_boundary_path: np.ndarray, mv_markers: data.MVMarkers
# ) -> base.ParameterizedPath:
#     parameterization_specs = spec.parameterization_specs["MV"]
#     start_ind_str = parameterization_specs["start"]
#     marker_ind_str = parameterization_specs["markers"]
#     marker_values = parameterization_specs["marker_values"]

#     start_ind = getattr(mv_markers, start_ind_str)
#     marker_inds = [getattr(mv_markers, m) for m in marker_ind_str]
#     reordered_path, relative_marker_inds = base.reorder_path_by_markers(
#         mv_boundary_path,
#         start_ind=start_ind,
#         marker_inds=marker_inds,
#         marker_values=marker_values,
#     )
#     parameterized_path = base.parameterize_path_by_relative_length(
#         mesh,
#         reordered_path,
#         relative_marker_inds=relative_marker_inds,
#         marker_values=marker_values,
#     )
#     return parameterized_path


# # --------------------------------------------------------------------------------------------------
# def parameterize_paths_without_markers(
#     mesh: pv.PolyData,
#     roof_paths: data.RoofPaths,
#     diagonal_paths: data.DiagonalPaths,
#     pv_segment_paths: data.PVSegmentPaths,
# ) -> data.ParameterizedRoofPaths:
#     parameterized_roof_paths = data.ParameterizedRoofPaths()
#     for roof_field in fields(parameterized_roof_paths):
#         roof_path = getattr(roof_paths, roof_field.name)
#         parameterized_path = base.parameterize_path_by_relative_length(mesh, roof_path)
#         setattr(parameterized_roof_paths, roof_field.name, parameterized_path)

#     parameterized_diagonal_paths = data.ParameterizedDiagonalPaths()
#     for diagonal_field in fields(diagonal_paths):
#         diagonal_path = getattr(diagonal_paths, diagonal_field.name)
#         parameterized_path = base.parameterize_path_by_relative_length(mesh, diagonal_path)
#         setattr(parameterized_diagonal_paths, diagonal_field.name, parameterized_path)

#     parameterized_pv_segment_paths = data.ParameterizedPVSegmentPaths()
#     for pv_field in fields(pv_segment_paths):
#         pv_segment = getattr(pv_segment_paths, pv_field.name)
#         parameterized_segments = data.ParameterizedSegments()
#         for segment_field in fields(parameterized_segments):
#             segment_path = getattr(pv_segment, segment_field.name)
#             parameterized_path = base.parameterize_path_by_relative_length(mesh, segment_path)
#             setattr(parameterized_segments, segment_field.name, parameterized_path)
#         setattr(parameterized_pv_segment_paths, pv_field.name, parameterized_segments)

#     return parameterized_roof_paths, parameterized_diagonal_paths, parameterized_pv_segment_paths


# # ==================================================================================================
# def construct_uac_pv_boundary_paths(
#     parameterized_pv_boundary_paths: data.ParameterizedPVBoundaryPaths,
#     pv_markers: data.PVSegmentMarkers,
# ) -> data.UACPVBoundaryPaths:
#     uac_pv_boundary_paths = data.UACPVBoundaryPaths()
#     uac_marker_specs = spec.uac_direct_marker_specs["pv_boundaries"]

#     for pv_boundary_field in fields(parameterized_pv_boundary_paths):
#         path = getattr(parameterized_pv_boundary_paths, pv_boundary_field.name)
#         marker_specs = uac_marker_specs[pv_boundary_field.name]
#         triangle = base.UACTriangle(
#             vertex_relative_lengths=(1 / 4, 5 / 8),
#             vertex_one=marker_specs["anterior_posterior"],
#             vertex_two=marker_specs["septal_lateral"],
#             vertex_three=marker_specs["diagonal"],
#         )
#         uac_path = base.compute_uacs_triangle(path, triangle)
#         setattr(uac_pv_boundary_paths, pv_boundary_field.name, uac_path)
#     return uac_pv_boundary_paths


# # --------------------------------------------------------------------------------------------------
# def construct_uac_roof_paths(
#     parameterized_roof_paths: data.ParameterizedRoofPaths,
# ) -> data.UACRoofPaths:
#     uac_roof_paths = data.UACRoofPaths()
#     uac_marker_specs = spec.uac_indirect_marker_specs["roof"]

#     for roof_field in fields(parameterized_roof_paths):
#         path = getattr(parameterized_roof_paths, roof_field.name)
#         start_point, end_point = uac_marker_specs[roof_field.name]
#         line = base.UACLine(start=start_point, end=end_point)
#         uac_path = base.compute_uacs_line(path, line)
#         setattr(uac_roof_paths, roof_field.name, uac_path)

#     return uac_roof_paths


# # --------------------------------------------------------------------------------------------------
# def construct_uac_diagonal_paths(
#     parameterized_diagonal_paths: data.ParameterizedDiagonalPaths,
# ) -> data.UACDiagonalPaths:
#     uac_diagonal_paths = data.UACDiagonalPaths()
#     uac_marker_specs = spec.uac_indirect_marker_specs["diagonal"]

#     for diagonal_field in fields(parameterized_diagonal_paths):
#         path = getattr(parameterized_diagonal_paths, diagonal_field.name)
#         start_point, end_point = uac_marker_specs[diagonal_field.name]
#         line = base.UACLine(start=start_point, end=end_point)
#         uac_path = base.compute_uacs_line(path, line)
#         setattr(uac_diagonal_paths, diagonal_field.name, uac_path)

#     return uac_diagonal_paths


# # --------------------------------------------------------------------------------------------------
# def construct_uac_pv_segment_paths(
#     parameterized_pv_segment_paths: data.ParameterizedPVSegmentPaths,
# ) -> data.UACPVSegmentPaths:
#     uac_pv_segment_paths = data.UACPVSegmentPaths()
#     uac_marker_specs = spec.uac_indirect_marker_specs["pv_segments"]

#     for pv_field in fields(parameterized_pv_segment_paths):
#         pv_segments = getattr(parameterized_pv_segment_paths, pv_field.name)
#         pv_specs = uac_marker_specs[pv_field.name]
#         segments = data.UACSegments()
#         for segment_field in fields(segments):
#             path = getattr(pv_segments, segment_field.name)
#             start_point, end_point = pv_specs[segment_field.name]
#             line = base.UACLine(start=start_point, end=end_point)
#             uac_path = base.compute_uacs_line(path, line)
#             setattr(segments, segment_field.name, uac_path)
#         setattr(uac_pv_segment_paths, pv_field.name, segments)
#     return uac_pv_segment_paths


# # --------------------------------------------------------------------------------------------------
# def construct_uac_laa_boundary_path(
#     parameterized_laa_boundary_path: base.ParameterizedPath,
# ) -> base.UACPath:
#     uac_marker_specs = spec.uac_direct_marker_specs["laa_boundary"]
#     lower_left = uac_marker_specs["LIPV"]
#     length_alpha = (uac_marker_specs["MV"][0] - uac_marker_specs["LIPV"][0]) / np.sqrt(2)
#     length_beta = (uac_marker_specs["MV"][1] - uac_marker_specs["LIPV"][1]) / np.sqrt(2)
#     rectangle = base.UACRectangle(
#         lower_left_corner=lower_left,
#         length_alpha=length_alpha,
#         length_beta=length_beta,
#     )
#     uac_laa_boundary_path = base.compute_uacs_rectangle(parameterized_laa_boundary_path, rectangle)
#     return uac_laa_boundary_path


# # --------------------------------------------------------------------------------------------------
# def construct_uac_mv_boundary_path(
#     parameterized_mv_boundary_path: base.ParameterizedPath,
# ) -> base.UACPath:
#     uac_marker_specs = spec.uac_direct_marker_specs["mv_boundary"]
#     lower_left = uac_marker_specs["RSPV"]
#     length_alpha = uac_marker_specs["LSPV"][0] - uac_marker_specs["RSPV"][0]
#     length_beta = uac_marker_specs["RIPV"][1] - uac_marker_specs["RSPV"][1]
#     rectangle = base.UACRectangle(
#         lower_left_corner=lower_left,
#         length_alpha=length_alpha,
#         length_beta=length_beta,
#     )
#     uac_mv_boundary_path = base.compute_uacs_rectangle(parameterized_mv_boundary_path, rectangle)
#     return uac_mv_boundary_path


# # --------------------------------------------------------------------------------------------------
# def get_patch_boundary_from_dict(uac_paths: data.UACPaths, patch: str) -> data.PatchBoundaries:
#     patch_boundary_specs = spec.patch_boundary_specs[patch]
#     ind_values = []
#     alpha_values = []
#     beta_values = []

#     ind_values, alpha_values, beta_values = _get_patch_boundaries_from_dict(
#         uac_paths, patch_boundary_specs, ind_values, alpha_values, beta_values
#     )
#     ind_values = np.concatenate(ind_values)
#     alpha_values = np.concatenate(alpha_values)
#     beta_values = np.concatenate(beta_values)
#     unique_inds, unique_mask = np.unique(ind_values, return_index=True)
#     uac_boundary_path = base.UACPath(
#         inds=unique_inds, alpha=alpha_values[unique_mask], beta=beta_values[unique_mask]
#     )

#     return uac_boundary_path


# # --------------------------------------------------------------------------------------------------
# def _get_patch_boundaries_from_dict(
#     paths: Any,  # noqa: ANN401
#     specs: Any,
#     ind_values: list,
#     alpha_values: list,
#     beta_values: list,
# ) -> tuple:
#     if isinstance(paths, base.UACPath):
#         relevant_path_section = np.where(
#             (paths.relative_lengths >= specs[0]) & (paths.relative_lengths <= specs[1])
#         )[0]
#         ind_values.append(paths.inds[relevant_path_section])
#         alpha_values.append(paths.alpha[relevant_path_section])
#         beta_values.append(paths.beta[relevant_path_section])
#     else:
#         for collection_name, collection_specs in specs.items():
#             collection = getattr(paths, collection_name)
#             ind_values, alpha_values, beta_values = _get_patch_boundaries_from_dict(
#                 collection, collection_specs, ind_values, alpha_values, beta_values
#             )
#     return ind_values, alpha_values, beta_values
