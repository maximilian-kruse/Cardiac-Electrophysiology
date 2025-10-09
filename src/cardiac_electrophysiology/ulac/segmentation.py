from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .base import segmentation_base as seg


# ==================================================================================================
@dataclass
class FeatureTags:
    MV: int = None
    LAA: int = None
    LIPV: int = None
    LSPV: int = None
    RIPV: int = None
    RSPV: int = None


@dataclass
class BoundaryPaths:
    LIPV_inner: np.ndarray = None
    LSPV_inner: np.ndarray = None
    RIPV_inner: np.ndarray = None
    RSPV_inner: np.ndarray = None
    LIPV_outer: np.ndarray = None
    LSPV_outer: np.ndarray = None
    RIPV_outer: np.ndarray = None
    RSPV_outer: np.ndarray = None
    LAA: np.ndarray = None
    MV: np.ndarray = None


# --------------------------------------------------------------------------------------------------
@dataclass
class PVConnectionPaths:
    anterior_posterior: np.ndarray = None
    septal_lateral: np.ndarray = None
    diagonal: np.ndarray = None


@dataclass
class ConnectionPaths:
    LIPV: PVConnectionPaths = None
    LSPV: PVConnectionPaths = None
    RIPV: PVConnectionPaths = None
    RSPV: PVConnectionPaths = None
    LIPV_LSPV: np.ndarray = None
    LSPV_RSPV: np.ndarray = None
    RSPV_RIPV: np.ndarray = None
    RIPV_LIPV: np.ndarray = None
    LIPV_LAA: np.ndarray = None
    LAA_MV: np.ndarray = None
    LSPV_MV: np.ndarray = None
    RIPV_MV: np.ndarray = None
    RSPV_MV: np.ndarray = None
    LAA_Posterior: np.ndarray = None


# --------------------------------------------------------------------------------------------------
@dataclass
class PVMarkers:
    anterior_posterior: float = None
    septal_lateral: float = None
    diagonal: float = None


@dataclass
class LAAMarkers:
    LIPV: float = None
    MV: float = None
    posterior: float = None


@dataclass
class MVMarkers:
    LAA: float = None
    LSPV: float = None
    RIPV: float = None
    RSPV: float = None


@dataclass
class Markers:
    LIPV_inner: PVMarkers = None
    LSPV_inner: PVMarkers = None
    RIPV_inner: PVMarkers = None
    RSPV_inner: PVMarkers = None
    LIPV_outer: PVMarkers = None
    LSPV_outer: PVMarkers = None
    RIPV_outer: PVMarkers = None
    RSPV_outer: PVMarkers = None
    LAA: LAAMarkers = None
    MV: MVMarkers = None


# --------------------------------------------------------------------------------------------------
@dataclass
class PVParameterizedConnectionPaths:
    anterior_posterior: seg.ParameterizedPath = None
    septal_lateral: seg.ParameterizedPath = None
    diagonal: seg.ParameterizedPath = None


@dataclass
class ParameterizedPaths:
    LIPV_connections: PVParameterizedConnectionPaths = None
    LSPV_connections: PVParameterizedConnectionPaths = None
    RIPV_connections: PVParameterizedConnectionPaths = None
    RSPV_connections: PVParameterizedConnectionPaths = None
    LIPV_inner: seg.ParameterizedPath = None
    LSPV_inner: seg.ParameterizedPath = None
    RIPV_inner: seg.ParameterizedPath = None
    RSPV_inner: seg.ParameterizedPath = None
    LIPV_outer: seg.ParameterizedPath = None
    LSPV_outer: seg.ParameterizedPath = None
    RIPV_outer: seg.ParameterizedPath = None
    RSPV_outer: seg.ParameterizedPath = None
    LAA: seg.ParameterizedPath = None
    MV: seg.ParameterizedPath = None
    LIPV_LAA: seg.ParameterizedPath = None
    LAA_MV: seg.ParameterizedPath = None
    LSPV_MV: seg.ParameterizedPath = None
    RIPV_MV: seg.ParameterizedPath = None
    RSPV_MV: seg.ParameterizedPath = None


# ==================================================================================================
roof_path_specs = {
    "LIPV_LSPV": ("LIPV_inner", "LSPV_inner"),
    "LSPV_RSPV": ("LSPV_inner", "RSPV_inner"),
    "RSPV_RIPV": ("RSPV_inner", "RIPV_inner"),
    "RIPV_LIPV": ("RIPV_inner", "LIPV_inner"),
}
diagonal_path_specs = {
    "LIPV_LAA": ("LIPV_inner", "LAA"),
    "LAA_MV": ("LAA", "MV"),
    "LSPV_MV": ("LSPV_inner", "MV"),
    "RIPV_MV": ("RIPV_inner", "MV"),
    "RSPV_MV": ("RSPV_inner", "MV"),
}
pv_inner_outer_path_specs = {
    "LIPV": {
        "anterior_posterior": ("LIPV_LSPV", 0),
        "septal_lateral": ("RIPV_LIPV", -1),
        "diagonal": ("LIPV_LAA", 0),
    },
    "LSPV": {
        "anterior_posterior": ("LIPV_LSPV", -1),
        "septal_lateral": ("LSPV_RSPV", 0),
        "diagonal": ("LSPV_MV", 0),
    },
    "RSPV": {
        "anterior_posterior": ("RSPV_RIPV", 0),
        "septal_lateral": ("LSPV_RSPV", -1),
        "diagonal": ("RSPV_MV", 0),
    },
    "RIPV": {
        "anterior_posterior": ("RSPV_RIPV", -1),
        "septal_lateral": ("RIPV_LIPV", 0),
        "diagonal": ("RIPV_MV", 0),
    },
}

marker_specs = {
    "PV": {
        "start": "anterior_posterior",
        "markers": ["septal_lateral", "diagonal"],
        "marker_values": [1 / 4, 2 / 3],
    },
    "LAA": {
        "start": "LIPV",
        "markers": ["MV", "posterior"],
        "marker_values": [1 / 2, 3 / 4],
    },
    "MV": {
        "start": "RSPV",
        "markers": ["LSPV", "LAA", "RIPV"],
        "marker_values": [1 / 4, 1 / 2, 3 / 4],
    },
}


# ==================================================================================================
def get_boundary_paths(mesh: pv.PolyData, feature_tags: FeatureTags) -> BoundaryPaths:
    boundary_paths = BoundaryPaths()
    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        tag_value = getattr(feature_tags, pv_name)
        inner_boundary, outer_boundary = seg.get_inner_outer_boundaries(mesh, tag_value)
        setattr(boundary_paths, f"{pv_name}_inner", inner_boundary)
        setattr(boundary_paths, f"{pv_name}_outer", outer_boundary)
    laa_boundary, _ = seg.get_inner_outer_boundaries(mesh, feature_tags.LAA)
    _, mv_boundary = seg.get_inner_outer_boundaries(mesh, feature_tags.MV)
    boundary_paths.LAA = laa_boundary
    boundary_paths.MV = mv_boundary
    return boundary_paths


# --------------------------------------------------------------------------------------------------
def construct_connection_paths(mesh: pv.PolyData, boundary_paths: BoundaryPaths) -> ConnectionPaths:
    connection_paths = ConnectionPaths()
    all_connections = {**roof_path_specs, **diagonal_path_specs}
    for path_name, (start_set_str, end_set_str) in all_connections.items():
        start_set = getattr(boundary_paths, start_set_str)
        end_set = getattr(boundary_paths, end_set_str)
        path = seg.construct_shortest_path_between_subsets(mesh, start_set, end_set)
        setattr(connection_paths, path_name, path)

    for pv_name, pv_specs in pv_inner_outer_path_specs.items():
        pv_paths = PVConnectionPaths()
        for path_name, (start_connection_str, start_ind) in pv_specs.items():
            start_connection = getattr(connection_paths, start_connection_str)
            start_set = np.array([start_connection[start_ind]])
            end_set = getattr(boundary_paths, f"{pv_name}_outer")
            path = seg.construct_shortest_path_between_subsets(mesh, start_set, end_set)
            setattr(pv_paths, path_name, path)
        setattr(connection_paths, pv_name, pv_paths)

    start_set = boundary_paths.LAA
    end_set = np.array([connection_paths.RIPV_MV[-1]])
    laa_posterior_path = seg.construct_shortest_path_between_subsets(mesh, start_set, end_set)
    connection_paths.LAA_Posterior = laa_posterior_path

    return connection_paths


# --------------------------------------------------------------------------------------------------
def get_markers(connection_paths: ConnectionPaths) -> Markers:
    markers = Markers()
    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        inner_markers = PVMarkers(
            anterior_posterior=getattr(connection_paths, pv_name).anterior_posterior[0],
            septal_lateral=getattr(connection_paths, pv_name).septal_lateral[0],
            diagonal=getattr(connection_paths, pv_name).diagonal[0],
        )
        outer_markers = PVMarkers(
            anterior_posterior=getattr(connection_paths, pv_name).anterior_posterior[-1],
            septal_lateral=getattr(connection_paths, pv_name).septal_lateral[-1],
            diagonal=getattr(connection_paths, pv_name).diagonal[-1],
        )
        setattr(markers, f"{pv_name}_inner", inner_markers)
        setattr(markers, f"{pv_name}_outer", outer_markers)
    markers.LAA = LAAMarkers(
        LIPV=connection_paths.LIPV_LAA[-1],
        MV=connection_paths.LAA_MV[0],
        posterior=connection_paths.LAA_Posterior[0],
    )
    markers.MV = MVMarkers(
        LAA=connection_paths.LAA_MV[-1],
        LSPV=connection_paths.LSPV_MV[-1],
        RIPV=connection_paths.RIPV_MV[-1],
        RSPV=connection_paths.RSPV_MV[-1],
    )
    return markers


# --------------------------------------------------------------------------------------------------
def parameterize_paths(
    mesh: pv.PolyData,
    boundary_paths: BoundaryPaths,
    connection_paths: ConnectionPaths,
    markers: Markers,
) -> ParameterizedPaths:
    parameterized_paths = ParameterizedPaths()
    parameterized_paths = _insert_parameterized_pv_boundary_paths(
        mesh, boundary_paths, markers, parameterized_paths
    )
    parameterized_paths = _insert_parameterized_pv_inner_outer_paths(
        mesh, connection_paths, parameterized_paths
    )
    parameterized_paths = _insert_parameterized_diagonal_paths(
        mesh, connection_paths, parameterized_paths
    )
    parameterized_paths = _insert_parameterized_laa_path(
        mesh, boundary_paths.LAA, markers.LAA, parameterized_paths
    )
    parameterized_paths = _insert_parameterized_mv_path(
        mesh, boundary_paths.MV, markers.MV, parameterized_paths
    )

    return parameterized_paths


# ==================================================================================================
def _insert_parameterized_pv_boundary_paths(
    mesh: pv.PolyData,
    boundary_paths: BoundaryPaths,
    markers: Markers,
    parameterized_paths: ParameterizedPaths,
) -> ParameterizedPaths:
    start_ind_str = marker_specs["PV"]["start"]
    marker_ind_str = marker_specs["PV"]["markers"]
    marker_values = marker_specs["PV"]["marker_values"]
    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        for boundary_type in ["inner", "outer"]:
            path = getattr(boundary_paths, f"{pv_name}_{boundary_type}")
            pv_markers = getattr(markers, f"{pv_name}_{boundary_type}")
            ordered_path, relative_marker_inds = seg.reorder_path_by_markers(
                path,
                start_ind=getattr(pv_markers, start_ind_str),
                marker_inds=[getattr(pv_markers, m) for m in marker_ind_str],
                marker_values=marker_values,
            )
            parameterized_path = seg.parameterize_path_by_relative_length(
                mesh,
                ordered_path,
                relative_marker_inds=relative_marker_inds,
                marker_values=marker_values,
            )
            setattr(parameterized_paths, f"{pv_name}_{boundary_type}", parameterized_path)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def _insert_parameterized_pv_inner_outer_paths(
    mesh: pv.PolyData, connection_paths: ConnectionPaths, parameterized_paths: ParameterizedPaths
) -> ParameterizedPaths:
    for pv_name in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        pv_paths = getattr(connection_paths, pv_name)
        pv_parameterized_paths = PVParameterizedConnectionPaths()
        for path_name in ["anterior_posterior", "septal_lateral", "diagonal"]:
            path = getattr(pv_paths, path_name)
            parameterized_path = seg.parameterize_path_by_relative_length(mesh, path)
            setattr(pv_parameterized_paths, path_name, parameterized_path)
        setattr(parameterized_paths, f"{pv_name}_connections", pv_parameterized_paths)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def _insert_parameterized_diagonal_paths(
    mesh: pv.PolyData, connection_paths: ConnectionPaths, parameterized_paths: ParameterizedPaths
) -> ParameterizedPaths:
    for path_name in ["LIPV_LAA", "LAA_MV", "LSPV_MV", "RIPV_MV", "RSPV_MV"]:
        path = getattr(connection_paths, path_name)
        parameterized_path = seg.parameterize_path_by_relative_length(mesh, path)
        setattr(parameterized_paths, path_name, parameterized_path)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def _insert_parameterized_laa_path(
    mesh: pv.PolyData,
    laa_path: np.ndarray,
    laa_markers: LAAMarkers,
    parameterized_paths: ParameterizedPaths,
) -> ParameterizedPaths:
    start_ind = getattr(laa_markers, marker_specs["LAA"]["start"])
    marker_inds = [getattr(laa_markers, m) for m in marker_specs["LAA"]["markers"]]
    marker_values = marker_specs["LAA"]["marker_values"]
    reordered_path, relative_marker_inds = seg.reorder_path_by_markers(
        laa_path,
        start_ind=start_ind,
        marker_inds=marker_inds,
        marker_values=marker_values,
    )
    parameterized_paths.LAA = seg.parameterize_path_by_relative_length(
        mesh,
        reordered_path,
        relative_marker_inds=[relative_marker_inds[0]],
        marker_values=[marker_values[0]],
    )
    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def _insert_parameterized_mv_path(
    mesh: pv.PolyData,
    mv_path: np.ndarray,
    mv_markers: MVMarkers,
    parameterized_paths: ParameterizedPaths,
) -> ParameterizedPaths:
    start_ind = getattr(mv_markers, marker_specs["MV"]["start"])
    marker_inds = [getattr(mv_markers, m) for m in marker_specs["MV"]["markers"]]
    marker_values = marker_specs["MV"]["marker_values"]
    reordered_path, relative_marker_inds = seg.reorder_path_by_markers(
        mv_path,
        start_ind=start_ind,
        marker_inds=marker_inds,
        marker_values=marker_values,
    )
    parameterized_paths.MV = seg.parameterize_path_by_relative_length(
        mesh,
        reordered_path,
        relative_marker_inds=relative_marker_inds,
        marker_values=marker_values,
    )
    return parameterized_paths
