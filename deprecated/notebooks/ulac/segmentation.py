from dataclasses import dataclass, fields

import numpy as np
import pyvista as pv

from . import segmentation_base as sb


# ==================================================================================================
@dataclass
class FeatureTags:
    MV: int=0
    LAA: int=1
    LIPV: int=2
    LSPV: int=3
    RIPV: int=4
    RSPV: int=5


# --------------------------------------------------------------------------------------------------
@dataclass
class Boundaries:
    LIPV_inner: sb.UnOrderedPath=None
    LSPV_inner: sb.UnOrderedPath=None
    RIPV_inner: sb.UnOrderedPath=None
    RSPV_inner: sb.UnOrderedPath=None
    LIPV_outer: sb.UnOrderedPath=None
    LSPV_outer: sb.UnOrderedPath=None
    RIPV_outer: sb.UnOrderedPath=None
    RSPV_outer: sb.UnOrderedPath=None
    LAA: sb.UnOrderedPath=None
    MV: sb.UnOrderedPath=None


@dataclass
class BoundaryPaths:
    LIPV_inner: np.ndarray=None
    LSPV_inner: np.ndarray=None
    RIPV_inner: np.ndarray=None
    RSPV_inner: np.ndarray=None
    LIPV_outer: np.ndarray=None
    LSPV_outer: np.ndarray=None
    RIPV_outer: np.ndarray=None
    RSPV_outer: np.ndarray=None
    LAA: np.ndarray=None
    MV: np.ndarray=None


# --------------------------------------------------------------------------------------------------
@dataclass
class RoofConnectionPaths:
    LIPV_to_LSPV: np.ndarray=None
    LIPV_to_RIPV: np.ndarray=None
    LSPV_to_RSPV: np.ndarray=None
    RIPV_to_RSPV: np.ndarray=None


@dataclass
class DiagonalMVConnectionPaths:
    LIPV_to_LAA: np.ndarray=None
    LAA_to_MV: np.ndarray=None
    LSPV_to_MV: np.ndarray=None
    RIPV_to_MV: np.ndarray=None
    RSPV_to_MV: np.ndarray=None


@dataclass
class PVInnerOuterPaths:
    anterior_to_posterior: np.ndarray=None
    septal_to_lateral: np.ndarray=None
    diagonal: np.ndarray=None


@dataclass
class PVAllInnerOuterPaths:
    LIPV: PVInnerOuterPaths=None
    LSPV: PVInnerOuterPaths=None
    RIPV: PVInnerOuterPaths=None
    RSPV: PVInnerOuterPaths=None


# --------------------------------------------------------------------------------------------------
@dataclass
class PVMarkers:
    anterior_posterior: int=0
    septal_lateral: int=0
    diagonal: int=0


@dataclass
class LAAMarkers:
    LIPV: int=0
    MV: int=0


@dataclass
class MVMarkers:
    LAA: int=0
    LSPV: int=0
    RIPV: int=0
    RSPV: int=0


@dataclass
class PVAllMarkers:
    LIPV_inner: PVMarkers=None
    LSPV_inner: PVMarkers=None
    RIPV_inner: PVMarkers=None
    RSPV_inner: PVMarkers=None
    LIPV_outer: PVMarkers=None
    LSPV_outer: PVMarkers=None
    RIPV_outer: PVMarkers=None
    RSPV_outer: PVMarkers=None


# ==================================================================================================
def extract_boundaries(mesh: pv.PolyData, feature_tags: FeatureTags) -> Boundaries:
    mesh_boundary_points = sb.get_boundary_point_coordinates(mesh)

    lipv_inner_boundary, lipv_outer_boundary = sb.get_inner_outer_boundaries(
        mesh, feature_tags.LIPV, mesh_boundary_points
    )
    lspv_inner_boundary, lspv_outer_boundary = sb.get_inner_outer_boundaries(
        mesh, feature_tags.LSPV, mesh_boundary_points
    )
    ripv_inner_boundary, ripv_outer_boundary = sb.get_inner_outer_boundaries(
        mesh, feature_tags.RIPV, mesh_boundary_points
    )
    rspv_inner_boundary, rspv_outer_boundary = sb.get_inner_outer_boundaries(
        mesh, feature_tags.RSPV, mesh_boundary_points
    )
    laa_boundary, _ = sb.get_inner_outer_boundaries(mesh, feature_tags.LAA, mesh_boundary_points)
    _, mv_boundary = sb.get_inner_outer_boundaries(mesh, feature_tags.MV, mesh_boundary_points)

    boundaries = Boundaries(
        LIPV_inner=lipv_inner_boundary,
        LSPV_inner=lspv_inner_boundary,
        RIPV_inner=ripv_inner_boundary,
        RSPV_inner=rspv_inner_boundary,
        LIPV_outer=lipv_outer_boundary,
        LSPV_outer=lspv_outer_boundary,
        RIPV_outer=ripv_outer_boundary,
        RSPV_outer=rspv_outer_boundary,
        LAA=laa_boundary,
        MV=mv_boundary,
    )

    return boundaries


# --------------------------------------------------------------------------------------------------
def construct_boundary_paths(boundaries: Boundaries) -> BoundaryPaths:
    boundary_paths = BoundaryPaths()

    for path_field in fields(BoundaryPaths):
        boundary = getattr(boundaries, path_field.name)
        ordered_path = sb.construct_path_from_boundary(boundary)
        setattr(boundary_paths, path_field.name, ordered_path)

    return boundary_paths


# --------------------------------------------------------------------------------------------------
def construct_roof_connection_paths(
    mesh: pv.PolyData, boundary_paths: BoundaryPaths
) -> RoofConnectionPaths:
    solver_setup = sb.prepare_eikonal_run(mesh)
    lipv_to_lspv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LIPV_inner, boundary_paths.LSPV_inner, solver_setup
    )
    lipv_to_ripv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LIPV_inner, boundary_paths.RIPV_inner, solver_setup
    )
    lspv_to_rspv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LSPV_inner, boundary_paths.RSPV_inner, solver_setup
    )
    ripv_to_rspv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.RIPV_inner, boundary_paths.RSPV_inner, solver_setup
    )

    roof_connection_paths = RoofConnectionPaths(
        LIPV_to_LSPV=lipv_to_lspv_path,
        LIPV_to_RIPV=lipv_to_ripv_path,
        LSPV_to_RSPV=lspv_to_rspv_path,
        RIPV_to_RSPV=ripv_to_rspv_path,
    )

    return roof_connection_paths


# --------------------------------------------------------------------------------------------------
def construct_diagonal_mv_connection_paths(
    mesh: pv.PolyData, boundary_paths: BoundaryPaths
) -> DiagonalMVConnectionPaths:
    solver_setup = sb.prepare_eikonal_run(mesh)
    lipv_to_laa_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LIPV_inner, boundary_paths.LAA, solver_setup
    )
    laa_to_mv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LAA, boundary_paths.MV, solver_setup
    )
    lspv_to_mv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.LSPV_inner, boundary_paths.MV, solver_setup
    )
    ripv_to_mv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.RIPV_inner, boundary_paths.MV, solver_setup
    )
    rspv_to_mv_path = sb.construct_shortest_path_between_boundaries(
        mesh, boundary_paths.RSPV_inner, boundary_paths.MV, solver_setup
    )

    diagonal_mv_connection_paths = DiagonalMVConnectionPaths(
        LIPV_to_LAA=lipv_to_laa_path,
        LAA_to_MV=laa_to_mv_path,
        LSPV_to_MV=lspv_to_mv_path,
        RIPV_to_MV=ripv_to_mv_path,
        RSPV_to_MV=rspv_to_mv_path,
    )

    return diagonal_mv_connection_paths


# --------------------------------------------------------------------------------------------------
def construct_pv_inner_outer_connection_paths(
    mesh: pv.PolyData,
    boundary_paths: BoundaryPaths,
    roof_connection_paths: RoofConnectionPaths,
    diagonal_mv_connection_paths: DiagonalMVConnectionPaths,
) -> PVAllInnerOuterPaths:
    solver_setup = sb.prepare_eikonal_run(mesh)

    anterior_posterior_point = roof_connection_paths.LIPV_to_LSPV[0]
    septal_lateral_point = roof_connection_paths.LIPV_to_RIPV[0]
    diagonal_point = diagonal_mv_connection_paths.LIPV_to_LAA[0]
    connection_points = (anterior_posterior_point, septal_lateral_point, diagonal_point)
    lipv_inner_outer_paths = _construct_inner_outer_connection_paths(
        mesh, boundary_paths.LIPV_outer, connection_points, solver_setup
    )

    anterior_posterior_point = roof_connection_paths.LIPV_to_LSPV[-1]
    septal_lateral_point = roof_connection_paths.LSPV_to_RSPV[0]
    diagonal_point = diagonal_mv_connection_paths.LSPV_to_MV[0]
    connection_points = (anterior_posterior_point, septal_lateral_point, diagonal_point)
    lspv_inner_outer_paths = _construct_inner_outer_connection_paths(
        mesh, boundary_paths.LSPV_outer, connection_points, solver_setup
    )

    anterior_posterior_point = roof_connection_paths.RIPV_to_RSPV[0]
    septal_lateral_point = roof_connection_paths.LIPV_to_RIPV[-1]
    diagonal_point = diagonal_mv_connection_paths.RIPV_to_MV[0]
    connection_points = (anterior_posterior_point, septal_lateral_point, diagonal_point)
    ripv_inner_outer_paths = _construct_inner_outer_connection_paths(
        mesh, boundary_paths.RIPV_outer, connection_points, solver_setup
    )

    anterior_posterior_point = roof_connection_paths.RIPV_to_RSPV[-1]
    septal_lateral_point = roof_connection_paths.LSPV_to_RSPV[-1]
    diagonal_point = diagonal_mv_connection_paths.RSPV_to_MV[0]
    connection_points = (anterior_posterior_point, septal_lateral_point, diagonal_point)
    rspv_inner_outer_paths = _construct_inner_outer_connection_paths(
        mesh, boundary_paths.RSPV_outer, connection_points, solver_setup
    )

    pv_inner_outer_paths = PVAllInnerOuterPaths(
        LIPV=lipv_inner_outer_paths,
        LSPV=lspv_inner_outer_paths,
        RIPV=ripv_inner_outer_paths,
        RSPV=rspv_inner_outer_paths,
    )

    return pv_inner_outer_paths


# --------------------------------------------------------------------------------------------------
def get_pv_markers(pv_inner_outer_connection_paths: PVAllInnerOuterPaths) -> PVAllMarkers:
    pv_markers = PVAllMarkers()

    for path_field in fields(PVAllInnerOuterPaths):
        connection_paths = getattr(pv_inner_outer_connection_paths, path_field.name)
        inner_markers = PVMarkers(
            anterior_posterior=connection_paths.anterior_to_posterior[0],
            septal_lateral=connection_paths.septal_to_lateral[0],
            diagonal=connection_paths.diagonal[0],
        )
        outer_markers = PVMarkers(
            anterior_posterior=connection_paths.anterior_to_posterior[-1],
            septal_lateral=connection_paths.septal_to_lateral[-1],
            diagonal=connection_paths.diagonal[-1],
        )
        setattr(pv_markers, f"{path_field.name}_inner", inner_markers)
        setattr(pv_markers, f"{path_field.name}_outer", outer_markers)

    return pv_markers


# --------------------------------------------------------------------------------------------------
def get_laa_markers(diagonal_mv_connection_paths: DiagonalMVConnectionPaths) -> LAAMarkers:
    laa_markers = LAAMarkers(
        LIPV=diagonal_mv_connection_paths.LIPV_to_LAA[-1],
        MV=diagonal_mv_connection_paths.LAA_to_MV[0],
    )
    return laa_markers


# --------------------------------------------------------------------------------------------------
def get_mv_markers(diagonal_mv_connection_paths: DiagonalMVConnectionPaths) -> MVMarkers:
    mv_markers = MVMarkers(
        LAA=diagonal_mv_connection_paths.LAA_to_MV[-1],
        LSPV=diagonal_mv_connection_paths.LSPV_to_MV[-1],
        RIPV=diagonal_mv_connection_paths.RIPV_to_MV[-1],
        RSPV=diagonal_mv_connection_paths.RSPV_to_MV[-1],
    )
    return mv_markers


# ==================================================================================================
def _construct_inner_outer_connection_paths(
    mesh: pv.PolyData,
    boundary_path: np.ndarray,
    connection_points: tuple[int, int, int],
    solver_setup: tuple,
) -> PVInnerOuterPaths:
    inner_to_outer_paths = []
    for point in connection_points:
        outer_to_inner_path = sb.construct_shortest_path_between_boundary_and_point(
            mesh, boundary_path, point, solver_setup
        )
        inner_to_outer_paths.append(np.flip(outer_to_inner_path))

    inner_outer_paths = PVInnerOuterPaths(
        anterior_to_posterior=inner_to_outer_paths[0],
        septal_to_lateral=inner_to_outer_paths[1],
        diagonal=inner_to_outer_paths[2],
    )
    return inner_outer_paths
