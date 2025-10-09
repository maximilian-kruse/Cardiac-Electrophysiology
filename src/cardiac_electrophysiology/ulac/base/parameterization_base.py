from dataclasses import dataclass

import numpy as np

from . import segmentation_base as seg


# ==================================================================================================
@dataclass
class UACPath:
    inds: np.ndarray = None
    alpha: np.ndarray = None
    beta: np.ndarray = None


@dataclass
class UACCircle:
    center: tuple[float, float] = (0, 0)
    radius: float = 0
    start_angle: float = 0
    orientation: float = 0


@dataclass
class UACRectangle:
    lower_left_corner: tuple[float, float] = (0, 0)
    length_alpha: float = 0
    length_beta: float = 0


@dataclass
class UACLine:
    start: tuple[float, float] = (0, 0)
    end: tuple[float, float] = (0, 0)


# ==================================================================================================
def compute_uacs_circle(path: seg.ParameterizedPath, uac_circle: UACCircle) -> UACPath:
    angle = uac_circle.start_angle + 2 * np.pi * uac_circle.orientation * path.relative_lengths
    alpha = uac_circle.center[0] + uac_circle.radius * np.cos(angle)
    beta = uac_circle.center[1] + uac_circle.radius * np.sin(angle)
    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path


# --------------------------------------------------------------------------------------------------
def compute_uacs_rectangle(path: seg.ParameterizedPath, uac_rectangle: UACRectangle) -> UACPath:
    first_edge = np.where(path.relative_lengths <= 0.25)[0]
    second_edge = np.where((path.relative_lengths > 0.25) & (path.relative_lengths < 0.5))[0]
    third_edge = np.where((path.relative_lengths >= 0.5) & (path.relative_lengths < 0.75))[0]
    fourth_edge = np.where(path.relative_lengths >= 0.75)[0]
    alpha = np.zeros(path.relative_lengths.size)
    beta = np.zeros(path.relative_lengths.size)

    alpha[first_edge] = 4 * uac_rectangle.length_alpha * path.relative_lengths[first_edge]
    alpha[second_edge] = uac_rectangle.length_alpha
    alpha[third_edge] = 4 * uac_rectangle.length_alpha * (0.75 - path.relative_lengths[third_edge])
    alpha[fourth_edge] = 0
    alpha += uac_rectangle.lower_left_corner[0]
    beta[first_edge] = 0
    beta[second_edge] = 4 * uac_rectangle.length_beta * (path.relative_lengths[second_edge] - 0.25)
    beta[third_edge] = uac_rectangle.length_beta
    beta[fourth_edge] = 4 * uac_rectangle.length_beta * (1 - path.relative_lengths[fourth_edge])
    beta += uac_rectangle.lower_left_corner[1]

    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path


# --------------------------------------------------------------------------------------------------
def compute_uacs_line(path: seg.ParameterizedPath, uac_line: UACLine) -> UACPath:
    alpha = uac_line.start[0] + (uac_line.end[0] - uac_line.start[0]) * path.relative_lengths
    beta = uac_line.start[1] + (uac_line.end[1] - uac_line.start[1]) * path.relative_lengths
    uac_path = UACPath(inds=path.inds, alpha=alpha, beta=beta)
    return uac_path
