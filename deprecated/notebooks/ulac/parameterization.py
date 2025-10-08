from dataclasses import dataclass, fields

import numpy as np
import pyvista as pv

from . import parameterization_base as par
from . import segmentation as seg


# ==================================================================================================
@dataclass
class PVParameterizedPaths:
    LIPV_inner: par.ParameterizedPath = None
    LIPV_outer: par.ParameterizedPath = None
    LSPV_inner: par.ParameterizedPath = None
    LSPV_outer: par.ParameterizedPath = None
    RIPV_inner: par.ParameterizedPath = None
    RIPV_outer: par.ParameterizedPath = None
    RSPV_inner: par.ParameterizedPath = None
    RSPV_outer: par.ParameterizedPath = None


@dataclass
class PVRoofCornerPaths:
    LIPV_corner: par.ParameterizedPath = None
    LSPV_corner: par.ParameterizedPath = None
    RIPV_corner: par.ParameterizedPath = None
    RSPV_corner: par.ParameterizedPath = None


@dataclass
class PVRoofConnectionPaths:
    LIPV_to_LSPV: par.ParameterizedPath = None
    LIPV_to_RIPV: par.ParameterizedPath = None
    LSPV_to_RSPV: par.ParameterizedPath = None
    RIPV_to_RSPV: par.ParameterizedPath = None


@dataclass
class PVInnerOuterPaths:
    anterior_to_posterior: par.ParameterizedPath = None
    septal_to_lateral: par.ParameterizedPath = None
    diagonal: par.ParameterizedPath = None


@dataclass
class PVAllInnerOuterPaths:
    LIPV: PVInnerOuterPaths = None
    LSPV: PVInnerOuterPaths = None
    RIPV: PVInnerOuterPaths = None
    RSPV: PVInnerOuterPaths = None

@dataclass
class PVSections:
    ap_to_sl: par.ParameterizedPath = None
    sl_to_diag: par.ParameterizedPath = None
    diag_to_ap: par.ParameterizedPath = None

@dataclass
class PVAllSections:
    LIPV_inner: PVSections = None
    LSPV_inner: PVSections = None
    RIPV_inner: PVSections = None
    RSPV_inner: PVSections = None
    LIPV_outer: PVSections = None
    LSPV_outer: PVSections = None
    RIPV_outer: PVSections = None
    RSPV_outer: PVSections = None


@dataclass
class PVInnerOuterPaths:
    anterior_to_posterior: np.ndarray=None
    septal_to_lateral: np.ndarray=None
    diagonal: np.ndarray=None


@dataclass
class PVUACBoundaries:
    LIPV_inner: par.UACPath = None
    LIPV_outer: par.UACPath = None
    LSPV_inner: par.UACPath = None
    LSPV_outer: par.UACPath = None
    RIPV_inner: par.UACPath = None
    RIPV_outer: par.UACPath = None
    RSPV_inner: par.UACPath = None
    RSPV_outer: par.UACPath = None


@dataclass
class RoofUACBoundaries:
    LIPV_corner: par.ParameterizedPath = None
    LSPV_corner: par.ParameterizedPath = None
    RIPV_corner: par.ParameterizedPath = None
    RSPV_corner: par.ParameterizedPath = None
    LIPV_to_LSPV: par.ParameterizedPath = None
    LIPV_to_RIPV: par.ParameterizedPath = None
    LSPV_to_RSPV: par.ParameterizedPath = None
    RIPV_to_RSPV: par.ParameterizedPath = None


@dataclass
class UACPVSpecs:
    RIPV_inner: par.UACCircle = None
    RIPV_outer: par.UACCircle = None
    RSPV_inner: par.UACCircle = None
    RSPV_outer: par.UACCircle = None
    LIPV_inner: par.UACCircle = None
    LIPV_outer: par.UACCircle = None
    LSPV_inner: par.UACCircle = None
    LSPV_outer: par.UACCircle = None


@dataclass
class UACRoofSpecs:
    LIPV_corner: par.UACLine = None
    LSPV_corner: par.UACLine = None
    RIPV_corner: par.UACLine = None
    RSPV_corner: par.UACLine = None
    LIPV_to_LSPV: par.UACLine = None
    LIPV_to_RIPV: par.UACLine = None
    LSPV_to_RSPV: par.UACLine = None
    RIPV_to_RSPV: par.UACLine = None

@dataclass
class UACInnerOuterSpecs:
    anterior_to_posterior: par.UACLine = None
    septal_to_lateral: par.UACLine = None
    diagonal: par.UACLine = None

@dataclass
class UACAllInnerOuterSpecs:
    LIPV: UACInnerOuterSpecs = None
    LSPV: UACInnerOuterSpecs = None
    RIPV: UACInnerOuterSpecs = None
    RSPV: UACInnerOuterSpecs = None


# ==================================================================================================
def get_uac_pv_specs(
    alpha_min: float = 0,
    alpha_max: float = 0,
    beta_min: float = 0,
    beta_max: float = 0,
    relative_center_lipv: tuple[float, float] = (0, 0),
    relative_center_lspv: tuple[float, float] = (0, 0),
    relative_center_ripv: tuple[float, float] = (0, 0),
    relative_center_rspv: tuple[float, float] = (0, 0),
    radius_pv_inner: float = 0,
    radius_pv_outer: float = 0,
):
    uac_pv_specs = UACPVSpecs()
    lipv_center = (
        relative_center_lipv[0] * (alpha_max - alpha_min) + alpha_min,
        relative_center_lipv[1] * (beta_max - beta_min) + beta_min,
    )
    lspv_center = (
        relative_center_lspv[0] * (alpha_max - alpha_min) + alpha_min,
        relative_center_lspv[1] * (beta_max - beta_min) + beta_min,
    )
    ripv_center = (
        relative_center_ripv[0] * (alpha_max - alpha_min) + alpha_min,
        relative_center_ripv[1] * (beta_max - beta_min) + beta_min,
    )
    rspv_center = (
        relative_center_rspv[0] * (alpha_max - alpha_min) + alpha_min,
        relative_center_rspv[1] * (beta_max - beta_min) + beta_min,
    )
    lipv_start_angle = 3 / 2 * np.pi
    lspv_start_angle = 1 / 2 * np.pi
    ripv_start_angle = 3 / 2 * np.pi
    rspv_start_angle = 1 / 2 * np.pi
    lipv_orientation = -1
    lspv_orientation = +1
    ripv_orientation = +1
    rspv_orientation = -1

    uac_pv_specs.LIPV_inner = par.UACCircle(
        center=lipv_center,
        radius=radius_pv_inner,
        start_angle=lipv_start_angle,
        orientation=lipv_orientation,
    )
    uac_pv_specs.LIPV_outer = par.UACCircle(
        center=lipv_center,
        radius=radius_pv_outer,
        start_angle=lipv_start_angle,
        orientation=lipv_orientation,
    )
    uac_pv_specs.LSPV_inner = par.UACCircle(
        center=lspv_center,
        radius=radius_pv_inner,
        start_angle=lspv_start_angle,
        orientation=lspv_orientation,
    )
    uac_pv_specs.LSPV_outer = par.UACCircle(
        center=lspv_center,
        radius=radius_pv_outer,
        start_angle=lspv_start_angle,
        orientation=lspv_orientation,
    )
    uac_pv_specs.RIPV_inner = par.UACCircle(
        center=ripv_center,
        radius=radius_pv_inner,
        start_angle=ripv_start_angle,
        orientation=ripv_orientation,
    )
    uac_pv_specs.RIPV_outer = par.UACCircle(
        center=ripv_center,
        radius=radius_pv_outer,
        start_angle=ripv_start_angle,
        orientation=ripv_orientation,
    )
    uac_pv_specs.RSPV_inner = par.UACCircle(
        center=rspv_center,
        radius=radius_pv_inner,
        start_angle=rspv_start_angle,
        orientation=rspv_orientation,
    )
    uac_pv_specs.RSPV_outer = par.UACCircle(
        center=rspv_center,
        radius=radius_pv_outer,
        start_angle=rspv_start_angle,
        orientation=rspv_orientation,
    )

    return uac_pv_specs


# --------------------------------------------------------------------------------------------------
def get_uac_laa_specs(
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
    relative_center_laa: tuple[float, float],
    radius_laa: float,
) -> par.UACRectangle:
    uac_laa_specs = par.UACCircle(
        center=(
            relative_center_laa[0] * (alpha_max - alpha_min) + alpha_min,
            relative_center_laa[1] * (beta_max - beta_min) + beta_min,
            5 / 6 * (beta_max - beta_min) + beta_min,
        ),
        radius=radius_laa,
        start_angle=5 / 4 * np.pi,
        orientation=+1,
    )
    return uac_laa_specs


# --------------------------------------------------------------------------------------------------
def get_uac_mv_specs(
    alpha_min: float,
    alpha_max: float,
    beta_min: float,
    beta_max: float,
) -> par.UACRectangle:
    uac_mv_specs = par.UACRectangle(
        lower_left_corner=(alpha_min, beta_min),
        length_alpha=alpha_max - alpha_min,
        length_beta=beta_max - beta_min,
    )
    return uac_mv_specs


# --------------------------------------------------------------------------------------------------
def get_uac_roof_specs(
    uac_pv_specs: UACPVSpecs,
) -> UACRoofSpecs:
    uac_roof_specs = UACRoofSpecs()
    for circle_field in fields(uac_pv_specs):
        if "outer" in circle_field.name:
            continue
        circle_spec = getattr(uac_pv_specs, circle_field.name)
        setattr(uac_roof_specs, circle_field.name.replace("inner", "corner"), circle_spec)

    uac_roof_specs.LIPV_to_LSPV = par.UACLine(
        start=(
            uac_pv_specs.LIPV_inner.center[0],
            uac_pv_specs.LIPV_inner.center[1] - uac_pv_specs.LIPV_inner.radius,
        ),
        end=(
            uac_pv_specs.LSPV_inner.center[0],
            uac_pv_specs.LSPV_inner.center[1] + uac_pv_specs.LSPV_inner.radius,
        ),
    )
    uac_roof_specs.LIPV_to_RIPV = par.UACLine(
        start=(
            uac_pv_specs.LIPV_inner.center[0] - uac_pv_specs.LIPV_inner.radius,
            uac_pv_specs.LIPV_inner.center[1],
        ),
        end=(
            uac_pv_specs.RIPV_inner.center[0] + uac_pv_specs.RIPV_inner.radius,
            uac_pv_specs.RIPV_inner.center[1],
        ),
    )
    uac_roof_specs.LSPV_to_RSPV = par.UACLine(
        start=(
            uac_pv_specs.LSPV_inner.center[0] - uac_pv_specs.LSPV_inner.radius,
            uac_pv_specs.LSPV_inner.center[1],
        ),
        end=(
            uac_pv_specs.RSPV_inner.center[0] + uac_pv_specs.RSPV_inner.radius,
            uac_pv_specs.RSPV_inner.center[1],
        ),
    )
    uac_roof_specs.RIPV_to_RSPV = par.UACLine(
        start=(
            uac_pv_specs.RIPV_inner.center[0],
            uac_pv_specs.RIPV_inner.center[1] - uac_pv_specs.RIPV_inner.radius,
        ),
        end=(
            uac_pv_specs.RSPV_inner.center[0],
            uac_pv_specs.RSPV_inner.center[1] + uac_pv_specs.RSPV_inner.radius,
        ),
    )

    return uac_roof_specs


# --------------------------------------------------------------------------------------------------
def get_uac_inner_outer_specs(
    uac_pv_specs: UACPVSpecs,
) -> UACAllInnerOuterSpecs:
    uac_inner_outer_specs = UACAllInnerOuterSpecs()
    for vein_field in fields(uac_inner_outer_specs):
        vein_specs = UACInnerOuterSpecs()
        circle_spec = getattr(uac_pv_specs, vein_field.name)
        inner_outer_spec = UACInnerOuterSpecs(
            anterior_to_posterior=par.UACLine(
                start=(
                    circle_spec.center[0],
                    circle_spec.center[1] - circle_spec.radius,
                ),
                end=(
                    circle_spec.center[0],
                    circle_spec.center[1] + circle_spec.radius,
                ),
            ),
            septal_to_lateral=par.UACLine(
                start=(
                    circle_spec.center[0] - circle_spec.radius,
                    circle_spec.center[1],
                ),
                end=(
                    circle_spec.center[0] + circle_spec.radius,
                    circle_spec.center[1],
                ),
            ),
            diagonal=par.UACLine(
                start=(
                    circle_spec.center[0] - circle_spec.radius / np.sqrt(2),
                    circle_spec.center[1] - circle_spec.radius / np.sqrt(2),
                ),
                end=(
                    circle_spec.center[0] + circle_spec.radius / np.sqrt(2),
                    circle_spec.center[1] + circle_spec.radius / np.sqrt(2),
                ),
            ),
        )
        setattr(uac_inner_outer_specs, circle_field.name.replace("inner", ""), inner_outer_spec)

    return uac_inner_outer_specs


# ==================================================================================================
def parameterize_pv_paths_by_arc_length(
    mesh: pv.PolyData,
    boundary_paths: seg.BoundaryPaths,
    pv_markers: seg.PVMarkers,
) -> PVParameterizedPaths:
    parameterized_paths = PVParameterizedPaths()
    for path in fields(parameterized_paths):
        path_id = path.name
        boundary_path = getattr(boundary_paths, path_id)
        pv_marker = getattr(pv_markers, path_id)
        parameterized_path = par.parameterize_polyline_by_path_length(
            mesh,
            boundary_path,
            start_ind=pv_marker.anterior_posterior,
            marker_inds=[pv_marker.septal_lateral, pv_marker.diagonal],
            marker_values=[1 / 4, 2 / 3],
        )
        setattr(parameterized_paths, path_id, parameterized_path)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def parameterize_laa_path_by_arc_length(
    mesh: pv.PolyData,
    boundary_paths: seg.BoundaryPaths,
    laa_markers: seg.LAAMarkers,
) -> par.ParameterizedPath:
    boundary_path = boundary_paths.LAA
    parameterized_path = par.parameterize_polyline_by_path_length(
        mesh,
        boundary_path,
        start_ind=laa_markers.LIPV,
        marker_inds=[laa_markers.MV],
        marker_values=[1 / 2],
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def parameterize_mv_path_by_arc_length(
    mesh: pv.PolyData,
    boundary_paths: seg.BoundaryPaths,
    mv_markers: seg.MVMarkers,
) -> par.ParameterizedPath:
    boundary_path = boundary_paths.MV
    parameterized_path = par.parameterize_polyline_by_path_length(
        mesh,
        boundary_path,
        start_ind=mv_markers.RSPV,
        marker_inds=[mv_markers.LSPV, mv_markers.LAA, mv_markers.RIPV],
        marker_values=[1 / 4, 1 / 2, 3 / 4],
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def parameterize_roof_connections(
    mesh: pv.PolyData, roof_connection_paths: seg.RoofConnectionPaths
):
    parameterized_paths = PVRoofConnectionPaths()
    for path in fields(parameterized_paths):
        path_id = path.name
        boundary_path = getattr(roof_connection_paths, path_id)
        parameterized_path = par.parameterize_polyline_by_path_length(
            mesh,
            boundary_path,
            start_ind=boundary_path[0],
            marker_inds=[],
            marker_values=[],
        )
        setattr(parameterized_paths, path_id, parameterized_path)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def parameterize_pv_inner_outer_paths(
    mesh: pv.PolyData, inner_outer_paths: seg.PVAllInnerOuterPaths
):
    parameterized_paths = PVAllInnerOuterPaths()
    for vein_field in fields(parameterized_paths):
        vein_id = vein_field.name
        vein_parameterized_paths = PVInnerOuterPaths()
        for path_field in fields(vein_parameterized_paths):
            path_id = path_field.name
            path = getattr(getattr(inner_outer_paths, vein_id), path_id)
            parameterized_path = par.parameterize_polyline_by_path_length(
                mesh,
                path,
                start_ind=path[0],
                marker_inds=[],
                marker_values=[],
            )
            setattr(vein_parameterized_paths, path_id, parameterized_path)
        setattr(parameterized_paths, vein_id, vein_parameterized_paths)

    return parameterized_paths


# --------------------------------------------------------------------------------------------------
def parameterize_pv_roof_corners(pv_parameterized_paths: PVParameterizedPaths):
    pv_roof_corner_paths = PVRoofCornerPaths()
    for corner in fields(pv_roof_corner_paths):
        pv_boundary_path = getattr(pv_parameterized_paths, corner.name.replace("_corner", "_inner"))
        corner_segment = np.where(pv_boundary_path.relative_length <= 0.25)[0]
        corner_path = par.ParameterizedPath(
            inds=pv_boundary_path.inds[corner_segment],
            relative_length=pv_boundary_path.relative_length[corner_segment],
        )
        setattr(pv_roof_corner_paths, corner.name, corner_path)

    return pv_roof_corner_paths


# --------------------------------------------------------------------------------------------------
def parameterize_pv_sections(pv_parameterized_paths: PVParameterizedPaths):
    pv_all_sections = PVAllSections()
    for pv_boundary_field in fields(pv_all_sections):
        pv_sections = PVSections()
        pv_boundary_path = getattr(pv_parameterized_paths, pv_boundary_field.name)
        segment_ap_to_sl = np.where(pv_boundary_path.relative_length <= 0.25)[0]
        pv_sections.ap_to_sl = par.ParameterizedPath(
            inds=pv_boundary_path.inds[segment_ap_to_sl],
            relative_length=pv_boundary_path.relative_length[segment_ap_to_sl],
        )
        segment_sl_to_diag = np.where(
            (pv_boundary_path.relative_length > 0.25) & (pv_boundary_path.relative_length <= 2 / 3)
        )[0]
        pv_sections.sl_to_diag = par.ParameterizedPath(
            inds=pv_boundary_path.inds[segment_sl_to_diag],
            relative_length=pv_boundary_path.relative_length[segment_sl_to_diag],
        )
        segment_diag_to_ap = np.where(pv_boundary_path.relative_length > 2 / 3)[0]
        pv_sections.diag_to_ap = par.ParameterizedPath(
            inds=pv_boundary_path.inds[segment_diag_to_ap],
            relative_length=pv_boundary_path.relative_length[segment_diag_to_ap],
        )
        setattr(pv_all_sections, pv_boundary_field.name, pv_sections)

    return pv_all_sections


# --------------------------------------------------------------------------------------------------
def compute_uac_pv_section_boundaries(
    parameterized_pv_paths: PVParameterizedPaths, uac_pv_specs: UACPVSpecs
) -> PVUACBoundaries:
    uac_boundaries = PVUACBoundaries()
    for path in fields(PVParameterizedPaths):
        path_id = path.name
        parameterized_path = getattr(parameterized_pv_paths, path_id)
        uac_circle = getattr(uac_pv_specs, path_id)
        uac_boundary = par.compute_uacs_circle(
            parameterized_path,
            uac_circle,
        )
        setattr(uac_boundaries, path_id, uac_boundary)
    return uac_boundaries


# --------------------------------------------------------------------------------------------------
def compute_uac_laa_boundary(
    parameterized_laa_path: par.ParameterizedPath, uac_specs: par.UACCircle
):
    uac_boundary = par.compute_uacs_circle(
        parameterized_laa_path,
        uac_specs,
    )
    return uac_boundary


# --------------------------------------------------------------------------------------------------
def compute_uac_mv_boundary(parameterized_mv_path: par.ParameterizedPath, uac_specs: par.UACCircle):
    uac_boundary = par.compute_uacs_rectangle(
        parameterized_mv_path,
        uac_specs,
    )
    return uac_boundary


# --------------------------------------------------------------------------------------------------
def compute_uac_roof_boundaries(
    roof_connection_paths: PVRoofConnectionPaths,
    roof_corner_paths: PVRoofCornerPaths,
    uac_roof_specs: UACRoofSpecs,
) -> PVParameterizedPaths:
    roof_boundaries = RoofUACBoundaries()
    for path in fields(roof_connection_paths):
        path_id = path.name
        parameterized_path = getattr(roof_connection_paths, path_id)
        uac_line = getattr(uac_roof_specs, path_id)
        uac_boundary = par.compute_uacs_line(
            parameterized_path,
            uac_line,
        )
        setattr(roof_boundaries, path_id, uac_boundary)

    for path in fields(roof_corner_paths):
        path_id = path.name
        boundary_path = getattr(roof_corner_paths, path_id)
        uac_circle = getattr(uac_roof_specs, path_id)
        uac_boundary = par.compute_uacs_circle(
            boundary_path,
            uac_circle,
        )
        setattr(roof_boundaries, path_id, uac_boundary)

    return roof_boundaries
