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
class UACCircle:
    center: float
    radius: float
    start_angle: float


@dataclass
class UACRectangle:
    length_alpha: float
    length_beta: float


@dataclass
class UACSpecs:
    RIPV_inner: UACCircle
    RIPV_outer: UACCircle
    RSPV_inner: UACCircle
    RSPV_outer: UACCircle
    LIPV_inner: UACCircle
    LIPV_outer: UACCircle
    LSPV_inner: UACCircle
    LSPV_outer: UACCircle
    LAA: UACCircle
    MV: UACRectangle

    def __init__(
        self,
        alpha_min: float = 0,
        alpha_max: float = 1,
        beta_min: float = 0,
        beta_max: float = 1,
        radius_pv_inner: float = 0.05,
        radius_pv_outer: float = 0.025,
        radius_laa: float = 0.05,
    ):
        self.RIPV_inner = UACCircle(
            center=1 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_inner,
            start_angle=3 / 2 * np.pi,
        )
        self.RIPV_outer = UACCircle(
            center=1 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_outer,
            start_angle=3 / 2 * np.pi,
        )
        self.RSPV_inner = UACCircle(
            center=1 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_inner,
            start_angle=0 * np.pi,
        )
        self.RSPV_outer = UACCircle(
            center=1 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_outer,
            start_angle=0 * np.pi,
        )
        self.LIPV_inner = UACCircle(
            center=2 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_inner,
            start_angle=1 * np.pi,
        )
        self.LIPV_outer = UACCircle(
            center=2 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_outer,
            start_angle=1 * np.pi,
        )
        self.LSPV_inner = UACCircle(
            center=2 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_inner,
            start_angle=1 / 2 * np.pi,
        )
        self.LSPV_outer = UACCircle(
            center=2 / 3 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_pv_outer,
            start_angle=1 / 2 * np.pi,
        )
        self.LAA = UACCircle(
            center=5 / 6 * (alpha_max - alpha_min) + alpha_min,
            radius=radius_laa,
            start_angle=3 / 2 * np.pi,
        )
        self.MV = UACRectangle(
            length_alpha=alpha_max - alpha_min,
            length_beta=beta_max - beta_min,
        )


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
        parameterized_path = par.parameterize_boundary_by_path_length(
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
    parameterized_path = par.parameterize_boundary_by_path_length(
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
    parameterized_path = par.parameterize_boundary_by_path_length(
        mesh,
        boundary_path,
        start_ind=mv_markers.RSPV,
        marker_inds=[mv_markers.LSPV, mv_markers.LAA, mv_markers.RIPV],
        marker_values=[1 / 4, 1 / 2, 3 / 4],
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def compute_alpha_circular(x: float, uac_circle: UACCircle):
    angle = uac_circle.start_angle + 2 * np.pi * x
    alpha = uac_circle.center + uac_circle.radius * np.cos(angle)
    return alpha


# --------------------------------------------------------------------------------------------------
def compute_beta_circular(y: float, uac_circle: UACCircle):
    angle = uac_circle.start_angle + 2 * np.pi * y
    beta = uac_circle.center + uac_circle.radius * np.sin(angle)
    return beta


# --------------------------------------------------------------------------------------------------
def compute_alpha_square(x: float, uac_rectangle: UACRectangle):
    alpha = np.zeros(x.size)
    first_edge = np.where(x <= 0.25)[0]
    second_edge = np.where((x > 0.25) & (x < 0.5))[0]
    third_edge = np.where((x >= 0.5) & (x < 0.75))[0]
    fourth_edge = np.where(x >= 0.75)[0]
    alpha[first_edge] = 4 * uac_rectangle.length_alpha * x[first_edge]
    alpha[second_edge] = uac_rectangle.length_alpha
    alpha[third_edge] = 4 * uac_rectangle.length_alpha * (0.75 - x[third_edge])
    alpha[fourth_edge] = 0
    return alpha


# --------------------------------------------------------------------------------------------------
def compute_beta_square(x, uac_rectangle: UACRectangle):
    beta = np.zeros(x.size)
    first_edge = np.where(x < 0.25)[0]
    second_edge = np.where((x >= 0.25) & (x < 0.5))[0]
    third_edge = np.where((x >= 0.5) & (x < 0.75))[0]
    fourth_edge = np.where(x >= 0.75)[0]
    beta[first_edge] = 0
    beta[second_edge] = 4 * uac_rectangle.length_beta * (x[second_edge] - 0.25)
    beta[third_edge] = uac_rectangle.length_beta
    beta[fourth_edge] = 4 * uac_rectangle.length_beta * (1 - x[fourth_edge])
    return beta
