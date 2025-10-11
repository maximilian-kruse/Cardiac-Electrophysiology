import copy
from dataclasses import fields
from typing import Any

import numpy as np
import pyvista as pv

from . import base
from . import data_structures as data
from . import specifications as spec


# --------------------------------------------------------------------------------------------------
def extract_boundary_paths(mesh: pv.PolyData, feature_tags: data.FeatureTags) -> data.Paths:
    pv_boundary_paths = data.PVBoundaryPaths()

    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        tag_value = getattr(feature_tags, pv_name)
        inner_boundary, outer_boundary = base.get_inner_outer_boundaries(mesh, tag_value)
        setattr(pv_boundary_paths, f"{pv_name}_inner", inner_boundary)
        setattr(pv_boundary_paths, f"{pv_name}_outer", outer_boundary)
    laa_boundary_path, _ = base.get_inner_outer_boundaries(mesh, feature_tags.LAA)
    _, mv_boundary_path = base.get_inner_outer_boundaries(mesh, feature_tags.MV)

    return pv_boundary_paths, laa_boundary_path, mv_boundary_path


# --------------------------------------------------------------------------------------------------
def construct_roof_paths(
    mesh: pv.PolyData, pv_boundary_paths: data.PVBoundaryPaths
) -> data.RoofPaths:
    roof_paths = data.RoofPaths()
    roof_specs = spec.path_specs["roof"]

    for path_name, (start_set_str, end_set_str) in roof_specs.items():
        start_set = getattr(pv_boundary_paths, start_set_str)
        end_set = getattr(pv_boundary_paths, end_set_str)
        path = base.construct_shortest_path_between_subsets(mesh, start_set, end_set)
        setattr(roof_paths, path_name, path)

    return roof_paths


# --------------------------------------------------------------------------------------------------
def construct_diagonal_paths(
    mesh: pv.PolyData,
    pv_boundary_paths: data.PVBoundaryPaths,
    laa_boundary_path: np.ndarray,
    mv_boundary_path: np.ndarray,
) -> data.DiagonalPaths:
    diagonal_paths = data.DiagonalPaths()
    diagonal_specs = spec.path_specs["diagonal"]

    combined_paths = copy.copy(pv_boundary_paths)
    combined_paths.LAA = laa_boundary_path
    combined_paths.MV = mv_boundary_path

    for path_name, (start_set_str, end_set_str) in diagonal_specs.items():
        start_set = getattr(combined_paths, start_set_str)
        end_set = getattr(combined_paths, end_set_str)
        path = base.construct_shortest_path_between_subsets(mesh, start_set, end_set)
        setattr(diagonal_paths, path_name, path)

    start_set = laa_boundary_path
    end_set = np.array([diagonal_paths.RIPV_MV[-1]])
    laa_posterior_path = base.construct_shortest_path_between_subsets(mesh, start_set, end_set)

    return diagonal_paths, laa_posterior_path


# --------------------------------------------------------------------------------------------------
def construct_pv_segment_paths(
    mesh: pv.PolyData,
    pv_boundary_paths: data.PVBoundaryPaths,
    roof_paths: data.RoofPaths,
    diagonal_paths: data.DiagonalPaths,
) -> data.PVSegmentPaths:
    pv_segment_paths = data.PVSegmentPaths()
    pv_segment_specs = spec.path_specs["pv_segments"]

    combined_paths = copy.copy(roof_paths)
    for diagonal_field in fields(diagonal_paths):
        setattr(combined_paths, diagonal_field.name, getattr(diagonal_paths, diagonal_field.name))

    for pv_name, pv_specs in pv_segment_specs.items():
        segments = data.Segments()
        for path_name, (start_connection_str, start_ind) in pv_specs.items():
            start_connection = getattr(combined_paths, start_connection_str)
            start_set = np.array([start_connection[start_ind]])
            end_set = getattr(pv_boundary_paths, f"{pv_name}_outer")
            path = base.construct_shortest_path_between_subsets(mesh, start_set, end_set)
            setattr(segments, path_name, path)
        setattr(pv_segment_paths, pv_name, segments)

    return pv_segment_paths


# ==================================================================================================
def parameterize_pv_boundary_paths(
    mesh: pv.PolyData,
    pv_boundary_paths: data.PVBoundaryPaths,
    pv_segment_markers: data.PVSegmentMarkers,
) -> data.ParameterizedPVBoundaryPaths:
    parameterized_pv_boundary_paths = data.ParameterizedPVBoundaryPaths()
    parameterization_specs = spec.parameterization_specs["PV"]
    start_ind_str = parameterization_specs["start"]
    marker_ind_str = parameterization_specs["markers"]
    marker_values = parameterization_specs["marker_values"]

    for pv_boundary_field in fields(parameterized_pv_boundary_paths):
        path = getattr(pv_boundary_paths, pv_boundary_field.name)
        path_markers = getattr(pv_segment_markers, pv_boundary_field.name)

        ordered_path, relative_marker_inds = base.reorder_path_by_markers(
            path,
            start_ind=getattr(path_markers, start_ind_str),
            marker_inds=[getattr(path_markers, m) for m in marker_ind_str],
            marker_values=marker_values,
        )
        parameterized_path = base.parameterize_path_by_relative_length(
            mesh,
            ordered_path,
            relative_marker_inds=relative_marker_inds,
            marker_values=marker_values,
        )
        setattr(parameterized_pv_boundary_paths, pv_boundary_field.name, parameterized_path)

    return parameterized_pv_boundary_paths


# --------------------------------------------------------------------------------------------------
def parameterize_laa_boundary_path(
    mesh: pv.PolyData, laa_boundary_path: np.ndarray, laa_markers: data.LAAMarkers
) -> base.ParameterizedPath:
    parameterization_specs = spec.parameterization_specs["LAA"]
    start_ind_str = parameterization_specs["start"]
    marker_ind_str = parameterization_specs["markers"]
    marker_values = parameterization_specs["marker_values"]

    start_ind = getattr(laa_markers, start_ind_str)
    marker_inds = [getattr(laa_markers, m) for m in marker_ind_str]
    reordered_path, relative_marker_inds = base.reorder_path_by_markers(
        laa_boundary_path,
        start_ind=start_ind,
        marker_inds=marker_inds,
        marker_values=marker_values,
    )
    parameterized_path = base.parameterize_path_by_relative_length(
        mesh,
        reordered_path,
        relative_marker_inds=[relative_marker_inds[0]],
        marker_values=[marker_values[0]],
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def parameterize_mv_boundary_path(
    mesh: pv.PolyData, mv_boundary_path: np.ndarray, mv_markers: data.MVMarkers
) -> base.ParameterizedPath:
    parameterization_specs = spec.parameterization_specs["MV"]
    start_ind_str = parameterization_specs["start"]
    marker_ind_str = parameterization_specs["markers"]
    marker_values = parameterization_specs["marker_values"]

    start_ind = getattr(mv_markers, start_ind_str)
    marker_inds = [getattr(mv_markers, m) for m in marker_ind_str]
    reordered_path, relative_marker_inds = base.reorder_path_by_markers(
        mv_boundary_path,
        start_ind=start_ind,
        marker_inds=marker_inds,
        marker_values=marker_values,
    )
    parameterized_path = base.parameterize_path_by_relative_length(
        mesh,
        reordered_path,
        relative_marker_inds=relative_marker_inds,
        marker_values=marker_values,
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def parameterize_paths_without_markers(
    mesh: pv.PolyData,
    roof_paths: data.RoofPaths,
    diagonal_paths: data.DiagonalPaths,
    pv_segment_paths: data.PVSegmentPaths,
) -> data.ParameterizedRoofPaths:
    parameterized_roof_paths = data.ParameterizedRoofPaths()
    for roof_field in fields(parameterized_roof_paths):
        roof_path = getattr(roof_paths, roof_field.name)
        parameterized_path = base.parameterize_path_by_relative_length(mesh, roof_path)
        setattr(parameterized_roof_paths, roof_field.name, parameterized_path)

    parameterized_diagonal_paths = data.ParameterizedDiagonalPaths()
    for diagonal_field in fields(diagonal_paths):
        diagonal_path = getattr(diagonal_paths, diagonal_field.name)
        parameterized_path = base.parameterize_path_by_relative_length(mesh, diagonal_path)
        setattr(parameterized_diagonal_paths, diagonal_field.name, parameterized_path)

    parameterized_pv_segment_paths = data.ParameterizedPVSegmentPaths()
    for pv_field in fields(pv_segment_paths):
        pv_segment = getattr(pv_segment_paths, pv_field.name)
        parameterized_segments = data.ParameterizedSegments()
        for segment_field in fields(parameterized_segments):
            segment_path = getattr(pv_segment, segment_field.name)
            parameterized_path = base.parameterize_path_by_relative_length(mesh, segment_path)
            setattr(parameterized_segments, segment_field.name, parameterized_path)
        setattr(parameterized_pv_segment_paths, pv_field.name, parameterized_segments)

    return parameterized_roof_paths, parameterized_diagonal_paths, parameterized_pv_segment_paths


# ==================================================================================================
def construct_uac_pv_boundary_paths(
    parameterized_pv_boundary_paths: data.ParameterizedPVBoundaryPaths,
) -> data.UACPVBoundaryPaths:
    uac_pv_boundary_paths = data.UACPVBoundaryPaths()
    uac_specs = spec.uac_form_specs["pv_boundaries"]
    for pv_boundary_field in fields(parameterized_pv_boundary_paths):
        path = getattr(parameterized_pv_boundary_paths, pv_boundary_field.name)
        circle = uac_specs[pv_boundary_field.name]
        uac_path = base.compute_uacs_circle(path, circle)
        setattr(uac_pv_boundary_paths, pv_boundary_field.name, uac_path)
    return uac_pv_boundary_paths


# --------------------------------------------------------------------------------------------------
def construct_uac_roof_paths(
    parameterized_roof_paths: data.ParameterizedRoofPaths,
) -> data.UACRoofPaths:
    uac_roof_paths = data.UACRoofPaths()
    uac_specs = spec.uac_form_specs["roof"]
    for roof_field in fields(parameterized_roof_paths):
        path = getattr(parameterized_roof_paths, roof_field.name)
        line = uac_specs[roof_field.name]
        uac_path = base.compute_uacs_line(path, line)
        setattr(uac_roof_paths, roof_field.name, uac_path)

    return uac_roof_paths


# --------------------------------------------------------------------------------------------------
def construct_uac_diagonal_paths(
    parameterized_diagonal_paths: data.ParameterizedDiagonalPaths,
) -> data.UACDiagonalPaths:
    uac_diagonal_paths = data.UACDiagonalPaths()
    uac_specs = spec.uac_form_specs["diagonal"]
    for diagonal_field in fields(parameterized_diagonal_paths):
        path = getattr(parameterized_diagonal_paths, diagonal_field.name)
        line = uac_specs[diagonal_field.name]
        uac_path = base.compute_uacs_line(path, line)
        setattr(uac_diagonal_paths, diagonal_field.name, uac_path)

    return uac_diagonal_paths


# --------------------------------------------------------------------------------------------------
def construct_uac_pv_segment_paths(
    parameterized_pv_segment_paths: data.ParameterizedPVSegmentPaths,
) -> data.UACPVSegmentPaths:
    uac_pv_segment_paths = data.UACPVSegmentPaths()
    uac_specs = spec.uac_form_specs["pv_segments"]
    for pv_field in fields(parameterized_pv_segment_paths):
        pv_segment = getattr(parameterized_pv_segment_paths, pv_field.name)
        uac_segments = data.UACSegments()
        for segment_field in fields(uac_segments):
            path = getattr(pv_segment, segment_field.name)
            line = uac_specs[pv_field.name][segment_field.name]
            uac_path = base.compute_uacs_line(path, line)
            setattr(uac_segments, segment_field.name, uac_path)
        setattr(uac_pv_segment_paths, pv_field.name, uac_segments)
    return uac_pv_segment_paths


# --------------------------------------------------------------------------------------------------
def get_patch_boundary_from_dict(uac_paths: data.UACPaths, patch: str) -> data.PatchBoundaries:
    patch_boundary_specs = spec.patch_boundary_specs[patch]
    ind_values = []
    alpha_values = []
    beta_values = []

    ind_values, alpha_values, beta_values = _get_patch_boundaries_from_dict(
        uac_paths, patch_boundary_specs, ind_values, alpha_values, beta_values
    )
    ind_values = np.concatenate(ind_values)
    alpha_values = np.concatenate(alpha_values)
    beta_values = np.concatenate(beta_values)
    unique_inds, unique_mask = np.unique(ind_values, return_index=True)
    uac_boundary_path = base.UACPath(
        inds=unique_inds, alpha=alpha_values[unique_mask], beta=beta_values[unique_mask]
    )

    return uac_boundary_path


# --------------------------------------------------------------------------------------------------
def _get_patch_boundaries_from_dict(
    paths: Any,  # noqa: ANN401
    specs: Any,
    ind_values: list,
    alpha_values: list,
    beta_values: list,
) -> tuple:
    if isinstance(paths, base.UACPath):
        relevant_path_section = np.where(
            (paths.relative_lengths >= specs[0]) & (paths.relative_lengths <= specs[1])
        )[0]
        ind_values.append(paths.inds[relevant_path_section])
        alpha_values.append(paths.alpha[relevant_path_section])
        beta_values.append(paths.beta[relevant_path_section])
    else:
        for collection_name, collection_specs in specs.items():
            collection = getattr(paths, collection_name)
            ind_values, alpha_values, beta_values = _get_patch_boundaries_from_dict(
                collection, collection_specs, ind_values, alpha_values, beta_values
            )
    return ind_values, alpha_values, beta_values
