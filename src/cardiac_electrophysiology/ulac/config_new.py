from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np


# ==================================================================================================
@dataclass
class MarkerConfig:
    path: Iterable[str]
    position_type: str
    position: int | float
    uacs: Iterable[Real]


@dataclass
class ParameterizationConfig:
    markers: Iterable[Iterable[str]]
    marker_relative_positions: Iterable[Real]


@dataclass
class BoundaryPathConfig:
    feature_tag: str
    coincides_with_mesh_boundary: bool
    parameterization: ParameterizationConfig


@dataclass
class ConnectionPathConfig:
    boundary_types: Iterable[str]
    start: Iterable[str]
    end: Iterable[str]
    parameterization: ParameterizationConfig
    inadmissible: Iterable[Iterable[str]] = None


# ==================================================================================================
path_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=False,
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "anterior_posterior"],
                    ["LIPV", "inner", "septal_lateral"],
                    ["LIPV", "inner", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=True,
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "outer", "anterior_posterior"],
                    ["LIPV", "outer", "septal_lateral"],
                    ["LIPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anterior_posterior"],
            end=["LIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "anterior_posterior"],
                    ["LIPV", "outer", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "septal_lateral"],
            end=["LIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "septal_lateral"],
                    ["LIPV", "outer", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anchor"],
            end=["LIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "anchor"],
                    ["LIPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=False,
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "anterior_posterior"],
                    ["LSPV", "inner", "septal_lateral"],
                    ["LSPV", "inner", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=True,
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "outer", "anterior_posterior"],
                    ["LSPV", "outer", "septal_lateral"],
                    ["LSPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anterior_posterior"],
            end=["LSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "anterior_posterior"],
                    ["LSPV", "outer", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "septal_lateral"],
            end=["LSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "septal_lateral"],
                    ["LSPV", "outer", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anchor"],
            end=["LSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["LSPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=False,
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "anterior_posterior"],
                    ["RSPV", "inner", "septal_lateral"],
                    ["RSPV", "inner", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=True,
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "outer", "anterior_posterior"],
                    ["RSPV", "outer", "septal_lateral"],
                    ["RSPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anterior_posterior"],
            end=["RSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "anterior_posterior"],
                    ["RSPV", "outer", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "septal_lateral"],
            end=["RSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "septal_lateral"],
                    ["RSPV", "outer", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anchor"],
            end=["RSPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "anchor"],
                    ["RSPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=False,
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "anterior_posterior"],
                    ["RIPV", "inner", "septal_lateral"],
                    ["RIPV", "inner", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=True,
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "outer", "anterior_posterior"],
                    ["RIPV", "outer", "septal_lateral"],
                    ["RIPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1 / 4, 5 / 8],
            ),
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anterior_posterior"],
            end=["RIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "anterior_posterior"],
                    ["RIPV", "outer", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "septal_lateral"],
            end=["RIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "septal_lateral"],
                    ["RIPV", "outer", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anchor"],
            end=["RIPV", "outer"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["RIPV", "outer", "anchor"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": BoundaryPathConfig(
        feature_tag="LAA",
        coincides_with_mesh_boundary=False,
        parameterization=ParameterizationConfig(
            markers=[["LAA", "LIPV"], ["LAA", "lateral"], ["LAA", "MV"], ["LAA", "posterior"]],
            marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
        ),
    ),
    # ----------------------------------------------------------------------------------------------
    "MV": BoundaryPathConfig(
        feature_tag="MV",
        coincides_with_mesh_boundary=True,
        parameterization=ParameterizationConfig(
            markers=[
                ["MV", "RSPV"],
                ["MV", "LSPV"],
                ["MV", "LAA"],
                ["MV", "RIPV"],
            ],
            marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
        ),
    ),
    # ----------------------------------------------------------------------------------------------
    "roof": {
        "LIPV_LSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LSPV", "inner"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "anterior_posterior"],
                    ["LSPV", "inner", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "LSPV_RSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["RSPV", "inner"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "septal_lateral"],
                    ["RSPV", "inner", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "RSPV_RIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["RIPV", "inner"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "anterior_posterior"],
                    ["RIPV", "inner", "anterior_posterior"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "RIPV_LIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["LIPV", "inner"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "septal_lateral"],
                    ["LIPV", "inner", "septal_lateral"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "anchor": {
        "LIPV_LAA": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LAA"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LIPV", "inner", "anchor"],
                    ["LAA", "LIPV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "LAA_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LAA"],
            end=["MV"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["LAA", "MV"],
                    ["MV", "LAA"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "LSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["MV"],
            inadmissible=(["LAA"], ["anchor", "LIPV_LAA"], ["anchor", "LAA_MV"]),
            parameterization=ParameterizationConfig(
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["MV", "LSPV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "RSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["MV"],
            parameterization=ParameterizationConfig(
                markers=[
                    ["RSPV", "inner", "anchor"],
                    ["MV", "RSPV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "RIPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["MV"],
            inadmissible=(["anchor", "RSPV_MV"],),
            parameterization=ParameterizationConfig(
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["MV", "RIPV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "LAA_lateral": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "LSPV_MV"],
            inadmissible=(["MV"],),
            parameterization=ParameterizationConfig(
                markers=[
                    ["LAA", "lateral"],
                    ["anchor", "LSPV_MV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
        "LAA_posterior": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "RIPV_MV"],
            inadmissible=(["MV"],),
            parameterization=ParameterizationConfig(
                markers=[
                    ["LAA", "posterior"],
                    ["anchor", "RIPV_MV"],
                ],
                marker_relative_positions=[0, 1],
            ),
        ),
    },
}


# ==================================================================================================
PV_INNER_RADIUS = 0.08
PV_OUTER_RADIUS = 0.04
LAA_RADIUS = 0.15
LIPV_CENTER = (2 / 3, 3 / 5)
LSPV_CENTER = (2 / 3, 2 / 5)
RIPV_CENTER = (1 / 3, 3 / 5)
RSPV_CENTER = (1 / 3, 2 / 5)
LAA_CENTER = (5 / 6, 5 / 6)
LAA_LATERAL_RATIO = 1 - (1 - LAA_CENTER[0]) / (1 - LIPV_CENTER[0])
LAA_POSTERIOR_RATIO = 1 - (1 - LAA_CENTER[1]) / (1 - LSPV_CENTER[1])


marker_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=0,
                uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=-1,
                uacs=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LIPV_LAA"],
                position_type="index",
                position=0,
                uacs=(LIPV_CENTER[0] + PV_INNER_RADIUS / 2, LIPV_CENTER[1] + PV_INNER_RADIUS / 2),
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
                uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["LIPV", "septal_lateral"],
                position_type="index",
                position=-1,
                uacs=(LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["LIPV", "anchor"],
                position_type="index",
                position=-1,
                uacs=(
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
    },
    "LSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=-1,
                uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=0,
                uacs=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LSPV_MV"],
                position_type="index",
                position=0,
                uacs=(
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
                uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["LSPV", "septal_lateral"],
                position_type="index",
                position=-1,
                uacs=(LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["LSPV", "anchor"],
                position_type="index",
                position=-1,
                uacs=(
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
    },
    "RSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=0,
                uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=-1,
                uacs=(RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RSPV_MV"],
                position_type="index",
                position=0,
                uacs=(
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
                uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["RSPV", "septal_lateral"],
                position_type="index",
                position=-1,
                uacs=(RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["RSPV", "anchor"],
                position_type="index",
                position=-1,
                uacs=(
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
    },
    "RIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=-1,
                uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=0,
                uacs=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RIPV_MV"],
                position_type="index",
                position=0,
                uacs=(
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
                uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ),
            "septal_lateral": MarkerConfig(
                path=["RIPV", "septal_lateral"],
                position_type="index",
                position=-1,
                uacs=(RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
            ),
            "anchor": MarkerConfig(
                path=["RIPV", "anchor"],
                position_type="index",
                position=-1,
                uacs=(
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": {
        "LIPV": MarkerConfig(
            path=["anchor", "LIPV_LAA"],
            position_type="index",
            position=-1,
            uacs=(LAA_CENTER[0] - LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2)),
        ),
        "MV": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=0,
            uacs=(LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2)),
        ),
        "lateral": MarkerConfig(
            path=["anchor", "LAA_lateral"],
            position_type="index",
            position=0,
            uacs=(LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2)),
        ),
        "posterior": MarkerConfig(
            path=["anchor", "LAA_posterior"],
            position_type="index",
            position=0,
            uacs=(LAA_CENTER[0] - LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2)),
        ),
    },
    "MV": {
        "RSPV": MarkerConfig(
            path=["anchor", "RSPV_MV"],
            position_type="index",
            position=-1,
            uacs=(0, 0),
        ),
        "LSPV": MarkerConfig(
            path=["anchor", "LSPV_MV"],
            position_type="index",
            position=-1,
            uacs=(1, 0),
        ),
        "LAA": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=-1,
            uacs=(1, 1),
        ),
        "RIPV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="index",
            position=-1,
            uacs=(0, 1),
        ),
    },
    "anchor": {
        "LSPV_MV": MarkerConfig(
            path=["anchor", "LSPV_MV"],
            position_type="relative",
            position=LAA_LATERAL_RATIO,
            uacs=(
                LSPV_CENTER[0] + LAA_LATERAL_RATIO * (1 - LSPV_CENTER[0]),
                LSPV_CENTER[1] - LAA_LATERAL_RATIO * (1 - LSPV_CENTER[1]),
            ),
        ),
        "RIPV_MV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="relative",
            position=LAA_POSTERIOR_RATIO,
            uacs=(
                RIPV_CENTER[0] - LAA_POSTERIOR_RATIO * (1 - RIPV_CENTER[0]),
                RIPV_CENTER[1] + LAA_POSTERIOR_RATIO * (1 - RIPV_CENTER[1]),
            ),
        ),
    },
}
