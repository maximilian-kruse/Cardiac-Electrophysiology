from dataclasses import dataclass, fields

import numpy as np

from .base import parameterization as par
from .base import segmentation as seg


# ==================================================================================================
@dataclass
class UACPaths:
    LIPV_inner: par.UACPath = None
    LSPV_inner: par.UACPath = None
    RIPV_inner: par.UACPath = None
    RSPV_inner: par.UACPath = None
    LIPV_outer: par.UACPath = None
    LSPV_outer: par.UACPath = None
    RIPV_outer: par.UACPath = None
    RSPV_outer: par.UACPath = None
    LAA: par.UACPath = None
    MV: par.UACPath = None


@dataclass
class UACConfig:
    alpha_min: float = None
    alpha_max: float = None
    beta_min: float = None
    beta_max: float = None
    radius_pv_inner: float = None
    radius_pv_outer: float = None
    radius_laa: float = None


@dataclass
class UACSpecs:
    LIPV_inner: par.UACCircle = None
    LSPV_inner: par.UACCircle = None
    RIPV_inner: par.UACCircle = None
    RSPV_inner: par.UACCircle = None
    LIPV_outer: par.UACCircle = None
    LSPV_outer: par.UACCircle = None
    RIPV_outer: par.UACCircle = None
    RSPV_outer: par.UACCircle = None
    LAA: par.UACCircle = None
    MV: par.UACRectangle = None


fixed_specs = {
    "LIPV": {
        "relative_center": (2 / 3, 2 / 3),
        "orientation": -1,
        "start_angle": 3 / 2 * np.pi,
    },
    "LSPV": {
        "relative_center": (2 / 3, 1 / 3),
        "orientation": +1,
        "start_angle": 1 / 2 * np.pi,
    },
    "RIPV": {
        "relative_center": (1 / 3, 2 / 3),
        "orientation": +1,
        "start_angle": 3 / 2 * np.pi,
    },
    "RSPV": {
        "relative_center": (1 / 3, 1 / 3),
        "orientation": -1,
        "start_angle": 1 / 2 * np.pi,
    },
    "LAA": {
        "relative_center": (5 / 6, 5 / 6),
        "orientation": +1,
        "start_angle": 5 / 4 * np.pi,
    },
}


# ==================================================================================================
def get_uac_specs(uac_config: UACConfig):
    uv_specs = UACSpecs()
    alpha_min = uac_config.alpha_min
    alpha_max = uac_config.alpha_max
    beta_min = uac_config.beta_min
    beta_max = uac_config.beta_max

    for pv_names in ["LIPV", "LSPV", "RIPV", "RSPV"]:
        start_angle = fixed_specs[pv_names]["start_angle"]
        orientation = fixed_specs[pv_names]["orientation"]
        relative_center = fixed_specs[pv_names]["relative_center"]
        center = (
            relative_center[0] * (alpha_max - alpha_min) + alpha_min,
            relative_center[1] * (beta_max - beta_min) + beta_min,
        )
        for boundary_type in ["inner", "outer"]:
            radius = getattr(uac_config, f"radius_pv_{boundary_type}")
            uac_circle = par.UACCircle(
                center=center,
                radius=radius,
                start_angle=start_angle,
                orientation=orientation,
            )
            setattr(uv_specs, f"{pv_names}_{boundary_type}", uac_circle)

    relative_center = fixed_specs["LAA"]["relative_center"]
    laa_center = (
            relative_center[0] * (alpha_max - alpha_min) + alpha_min,
            relative_center[1] * (beta_max - beta_min) + beta_min,
        )
    uv_specs.LAA = par.UACCircle(
        center=laa_center,
        radius=uac_config.radius_laa,
        start_angle=fixed_specs["LAA"]["start_angle"],
        orientation=fixed_specs["LAA"]["orientation"],
    )

    uv_specs.MV = par.UACRectangle(
        lower_left_corner=(alpha_min, beta_min),
        length_alpha=alpha_max - alpha_min,
        length_beta=beta_max - beta_min,
    )

    return uv_specs


# --------------------------------------------------------------------------------------------------
def compute_uac_paths(parameterized_paths: seg.ParameterizedPath, uac_specs: UACSpecs) -> UACPaths:
    uac_paths = UACPaths()
    for field in fields(uac_paths):
        path = getattr(parameterized_paths, field.name)
        spec = getattr(uac_specs, field.name)
        if isinstance(spec, par.UACCircle):
            uac_path = par.compute_uacs_circle(path, spec)
        elif isinstance(spec, par.UACRectangle):
            uac_path = par.compute_uacs_rectangle(path, spec)
        setattr(uac_paths, field.name, uac_path)
    return uac_paths


# --------------------------------------------------------------------------------------------------
def extract_boundary_data(*uac_paths: UACPaths):
    boundary_indices = []
    boundary_values = []

    for uac_path in uac_paths:
        boundary_indices.append(uac_path.inds)
        boundary_values.append(np.vstack([uac_path.alpha, uac_path.beta]).T)

    boundary_indices = np.concatenate(boundary_indices)
    boundary_values = np.vstack(boundary_values)
    _, unique_mask = np.unique(boundary_indices, axis=0, return_index=True)
    boundary_indices = boundary_indices[unique_mask]
    boundary_values = boundary_values[unique_mask]

    return boundary_indices, boundary_values
