import numpy as np
import pyvista as pv

from . import base
from . import interface as comp
from . import data_structures_old as data
from . import specifications as spec

FeatureTags = data.FeatureTags


# ==================================================================================================
def construct_paths(mesh: pv.PolyData, feature_tags: FeatureTags) -> data.Paths:
    pv_boundary_paths, laa_boundary_path, mv_boundary_path = comp.extract_boundary_paths(
        mesh, feature_tags
    )
    roof_paths = comp.construct_roof_paths(mesh, pv_boundary_paths)
    diagonal_paths, laa_posterior_path = comp.construct_diagonal_paths(
        mesh, pv_boundary_paths, laa_boundary_path, mv_boundary_path
    )
    pv_segment_paths = comp.construct_pv_segment_paths(
        mesh, pv_boundary_paths, roof_paths, diagonal_paths
    )
    paths = data.Paths(
        pv_boundaries=pv_boundary_paths,
        laa_boundary=laa_boundary_path,
        mv_boundary=mv_boundary_path,
        pv_segments=pv_segment_paths,
        roof=roof_paths,
        diagonal=diagonal_paths,
        laa_posterior=laa_posterior_path,
    )
    return paths


# --------------------------------------------------------------------------------------------------
def get_marker_points(paths: data.Paths) -> data.Markers:
    pv_segment_paths = paths.pv_segments
    pv_segment_markers = data.PVSegmentMarkers()
    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        for i, boundary_type in zip([0, -1], ["inner", "outer"], strict=True):
            path_markers = data.SegmentMarkers(
                anterior_posterior=getattr(pv_segment_paths, pv_name).anterior_posterior[i],
                septal_lateral=getattr(pv_segment_paths, pv_name).septal_lateral[i],
                diagonal=getattr(pv_segment_paths, pv_name).diagonal[i],
            )
            setattr(pv_segment_markers, f"{pv_name}_{boundary_type}", path_markers)

    diagonal_paths = paths.diagonal
    laa_markers = data.LAAMarkers(
        LIPV=diagonal_paths.LIPV_LAA[-1],
        MV=diagonal_paths.LAA_MV[0],
        posterior=paths.laa_posterior[0],
    )
    mv_markers = data.MVMarkers(
        LAA=diagonal_paths.LAA_MV[-1],
        LSPV=diagonal_paths.LSPV_MV[-1],
        RIPV=diagonal_paths.RIPV_MV[-1],
        RSPV=diagonal_paths.RSPV_MV[-1],
    )

    markers = data.Markers(pv_segments=pv_segment_markers, laa=laa_markers, mv=mv_markers)
    return markers


# --------------------------------------------------------------------------------------------------
def parameterize_paths(
    mesh: pv.PolyData, paths: data.Paths, markers: data.Markers
) -> data.ParameterizedPaths:
    parameterized_pv_boundary_paths = comp.parameterize_pv_boundary_paths(
        mesh, paths.pv_boundaries, markers.pv_segments
    )
    parameterized_laa_boundary_path = comp.parameterize_laa_boundary_path(
        mesh, paths.laa_boundary, markers.laa
    )
    parameterized_mv_boundary_path = comp.parameterize_mv_boundary_path(
        mesh, paths.mv_boundary, markers.mv
    )
    parameterized_roof_paths, parameterized_diagonal_paths, parameterized_pv_segment_paths = (
        comp.parameterize_paths_without_markers(mesh, paths.roof, paths.diagonal, paths.pv_segments)
    )

    parameterize_paths = data.ParameterizedPaths(
        pv_boundaries=parameterized_pv_boundary_paths,
        laa_boundary=parameterized_laa_boundary_path,
        mv_boundary=parameterized_mv_boundary_path,
        roof=parameterized_roof_paths,
        diagonal=parameterized_diagonal_paths,
        pv_segments=parameterized_pv_segment_paths,
    )
    return parameterize_paths


# --------------------------------------------------------------------------------------------------
def construct_uac_boundaries(
    parameterized_paths: data.ParameterizedPaths, markers: data.Markers
) -> data.UACPaths:
    uac_pv_boundary_paths = comp.construct_uac_pv_boundary_paths(
        parameterized_paths.pv_boundaries, markers.pv_segments
    )
    uac_roof_paths = comp.construct_uac_roof_paths(parameterized_paths.roof)
    uac_diagonal_paths = comp.construct_uac_diagonal_paths(parameterized_paths.diagonal)
    uac_pv_segment_paths = comp.construct_uac_pv_segment_paths(parameterized_paths.pv_segments)
    uac_laa_boundary_path = comp.construct_uac_laa_boundary_path(parameterized_paths.laa_boundary)
    uac_mv_boundary_path = comp.construct_uac_mv_boundary_path(parameterized_paths.mv_boundary)

    uac_paths = data.UACPaths(
        pv_boundaries=uac_pv_boundary_paths,
        laa_boundary=uac_laa_boundary_path,
        mv_boundary=uac_mv_boundary_path,
        roof=uac_roof_paths,
        diagonal=uac_diagonal_paths,
        pv_segments=uac_pv_segment_paths,
    )
    return uac_paths


# --------------------------------------------------------------------------------------------------
def get_patch_boundaries(uac_paths: data.UACPaths) -> data.PatchBoundaries:
    roof_patch_boundary = comp.get_patch_boundary_from_dict(uac_paths, "roof")
    anterior_patch_boundary = comp.get_patch_boundary_from_dict(uac_paths, "anterior")
    posterior_patch_boundary = comp.get_patch_boundary_from_dict(uac_paths, "posterior")
    septal_patch_boundary = comp.get_patch_boundary_from_dict(uac_paths, "septal")
    lateral_patch_boundary = comp.get_patch_boundary_from_dict(uac_paths, "lateral")
    patch_boundaries = data.PatchBoundaries(
        roof=roof_patch_boundary,
        laa=uac_paths.laa_boundary,
        anterior=anterior_patch_boundary,
        posterior=posterior_patch_boundary,
        septal=septal_patch_boundary,
        lateral=lateral_patch_boundary,
    )
    return patch_boundaries


# --------------------------------------------------------------------------------------------------
def extract_patches(patch_boundaries: data.PatchBoundaries) -> data.Patches:
    pass


# --------------------------------------------------------------------------------------------------
def compute_uac_coordinates(
    patch_boundaries: data.PatchBoundaries, patches: data.Patches
) -> np.ndarray:
    pass
