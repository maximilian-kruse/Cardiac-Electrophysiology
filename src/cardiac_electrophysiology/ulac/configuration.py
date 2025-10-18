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


@dataclass
class ParameterizationConfig:
    path: Iterable[str]
    markers: Iterable[Iterable[str]]
    marker_relative_positions: Iterable[Real]


@dataclass
class BoundaryPathConfig:
    feature_tag: str
    coincides_with_mesh_boundary: bool


@dataclass
class ConnectionPathConfig:
    boundary_types: Iterable[str]
    start: Iterable[str]
    end: Iterable[str]
    inadmissible_contact: Iterable[Iterable[str]]
    inadmissible_along: Iterable[Iterable[str]]


@dataclass
class UACConfig:
    path: Iterable[str]
    markers: Iterable[Iterable[float]]
    uacs: Iterable[tuple[float, float]]


@dataclass
class SubmeshConfig:
    boundary_paths: Iterable[Iterable[str]]
    outside_path: Iterable[str]


# ==================================================================================================
path_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anterior_posterior"],
            end=["LIPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "septal_lateral"],
            end=["LIPV", "outer"],
            inadmissible_contact=[["LIPV", "anterior_posterior"]],
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anchor"],
            end=["LIPV", "outer"],
            inadmissible_contact=[["LIPV", "anterior_posterior"], ["LIPV", "septal_lateral"]],
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anterior_posterior"],
            end=["LSPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "septal_lateral"],
            end=["LSPV", "outer"],
            inadmissible_contact=[["LSPV", "anterior_posterior"]],
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anchor"],
            end=["LSPV", "outer"],
            inadmissible_contact=[["LSPV", "anterior_posterior"], ["LSPV", "septal_lateral"]],
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anterior_posterior"],
            end=["RSPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "septal_lateral"],
            end=["RSPV", "outer"],
            inadmissible_contact=[["RSPV", "anterior_posterior"]],
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anchor"],
            end=["RSPV", "outer"],
            inadmissible_contact=[["RSPV", "anterior_posterior"], ["RSPV", "septal_lateral"]],
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anterior_posterior"],
            end=["RIPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "septal_lateral"],
            end=["RIPV", "outer"],
            inadmissible_contact=[["RIPV", "anterior_posterior"]],
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anchor"],
            end=["RIPV", "outer"],
            inadmissible_contact=[["RIPV", "anterior_posterior"], ["RIPV", "septal_lateral"]],
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": BoundaryPathConfig(
        feature_tag="LAA",
        coincides_with_mesh_boundary=False,
    ),
    # ----------------------------------------------------------------------------------------------
    "MV": BoundaryPathConfig(
        feature_tag="MV",
        coincides_with_mesh_boundary=True,
    ),
    # ----------------------------------------------------------------------------------------------
    "roof": {
        "LIPV_LSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LSPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LSPV", "inner"]],
        ),
        "LSPV_RSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["RSPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["LSPV", "inner"], ["RSPV", "inner"]],
        ),
        "RSPV_RIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["RIPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["RIPV", "inner"]],
        ),
        "RIPV_LIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["LIPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["RIPV", "inner"], ["LIPV", "inner"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "anchor": {
        "LIPV_LAA": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LAA"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LAA"]],
        ),
        "LAA_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LAA"],
            end=["MV"],
            inadmissible_contact=None,
            inadmissible_along=[["LAA"], ["MV"]],
        ),
        "LSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["MV"],
            inadmissible_contact=(["LAA"], ["anchor", "LIPV_LAA"], ["anchor", "LAA_MV"]),
            inadmissible_along=[["LSPV", "inner"], ["MV"]],
        ),
        "RSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["MV"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["MV"]],
        ),
        "RIPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["MV"],
            inadmissible_contact=(["anchor", "RSPV_MV"],),
            inadmissible_along=[["RIPV", "inner"], ["MV"]],
        ),
        "LAA_lateral": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "LSPV_MV"],
            inadmissible_contact=(["MV"],),
            inadmissible_along=[["LAA"], ["anchor", "LSPV_MV"]],
        ),
        "LAA_posterior": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "RIPV_MV"],
            inadmissible_contact=(["MV"],),
            inadmissible_along=[["LAA"], ["anchor", "RIPV_MV"]],
        ),
    },
}


# ==================================================================================================
parameterization_configs = {
    "anchor": {
        "LSPV_MV": ParameterizationConfig(
            path=["anchor", "LSPV_MV"],
            markers=[
                ["LSPV", "inner", "anchor"],
                ["MV", "LSPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_MV": ParameterizationConfig(
            path=["anchor", "RIPV_MV"],
            markers=[
                ["RIPV", "inner", "anchor"],
                ["MV", "RIPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
    }
}


# ==================================================================================================
marker_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=0,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LIPV_LAA"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["LIPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["LIPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "LSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=0,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LSPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["LSPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["LSPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "RSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=0,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RSPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["RSPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["RSPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "RIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=0,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RIPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["RIPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["RIPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": {
        "LIPV": MarkerConfig(
            path=["anchor", "LIPV_LAA"],
            position_type="index",
            position=-1,
        ),
        "MV": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=0,
        ),
        "lateral": MarkerConfig(
            path=["anchor", "LAA_lateral"],
            position_type="index",
            position=0,
        ),
        "posterior": MarkerConfig(
            path=["anchor", "LAA_posterior"],
            position_type="index",
            position=0,
        ),
    },
    "MV": {
        "RSPV": MarkerConfig(
            path=["anchor", "RSPV_MV"],
            position_type="index",
            position=-1,
        ),
        "LSPV": MarkerConfig(
            path=["anchor", "LSPV_MV"],
            position_type="index",
            position=-1,
        ),
        "LAA": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=-1,
        ),
        "RIPV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="index",
            position=-1,
        ),
    },
    "anchor": {
        "LSPV_MV": MarkerConfig(
            path=["anchor", "LSPV_MV"],
            position_type="relative",
            position=0.5,
        ),
        "RIPV_MV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="relative",
            position=0.5,
        ),
    },
}


# ==================================================================================================
LIPV_CENTER = (2, 3 / 2)
LSPV_CENTER = (2, 1)
RIPV_CENTER = (1, 3 / 2)
RSPV_CENTER = (1, 1)
LAA_CENTER = (5 / 2, 2)
PV_INNER_RADIUS = 1 / 5
PV_OUTER_RADIUS = 1 / 10
LAA_RADIUS = 1 / 2
LAA_DISTANCE = 1 / 4
LAA_DEPTH = 1 / 5
ANCHOR_LENGTH = 1


# ==================================================================================================
uac_configs = {
    "LIPV": {
        "inner": {
            "segment_1": UACConfig(
                path=["LIPV", "inner"],
                markers=[
                    ["LIPV", "inner", "anterior_posterior"],
                    ["LIPV", "inner", "septal_lateral"],
                ],
                uacs=[
                    (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                    (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["LIPV", "inner"],
                markers=[
                    ["LIPV", "inner", "septal_lateral"],
                    ["LIPV", "inner", "anchor"],
                ],
                uacs=[
                    (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["LIPV", "inner"],
                markers=[
                    ["LIPV", "inner", "anchor"],
                    ["LIPV", "inner", "anterior_posterior"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                ],
            ),
        },
        "outer": {
            "segment_1": UACConfig(
                path=["LIPV", "outer"],
                markers=[
                    ["LIPV", "outer", "anterior_posterior"],
                    ["LIPV", "outer", "septal_lateral"],
                ],
                uacs=[
                    (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
                    (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["LIPV", "outer"],
                markers=[
                    ["LIPV", "outer", "septal_lateral"],
                    ["LIPV", "outer", "anchor"],
                ],
                uacs=[
                    (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
                    (
                        LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["LIPV", "outer"],
                markers=[
                    ["LIPV", "outer", "anchor"],
                    ["LIPV", "outer", "anterior_posterior"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                    (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
                ],
            ),
        },
        "anterior_posterior": UACConfig(
            path=["LIPV", "anterior_posterior"],
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LIPV", "outer", "anterior_posterior"],
            ],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["LIPV", "septal_lateral"],
            markers=[
                ["LIPV", "inner", "septal_lateral"],
                ["LIPV", "outer", "septal_lateral"],
            ],
            uacs=[
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["LIPV", "anchor"],
            markers=[
                ["LIPV", "inner", "anchor"],
                ["LIPV", "outer", "anchor"],
            ],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "LSPV": {
        "inner": {
            "segment_1": UACConfig(
                path=["LSPV", "inner"],
                markers=[
                    ["LSPV", "inner", "anterior_posterior"],
                    ["LSPV", "inner", "septal_lateral"],
                ],
                uacs=[
                    (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                    (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["LSPV", "inner"],
                markers=[
                    ["LSPV", "inner", "septal_lateral"],
                    ["LSPV", "inner", "anchor"],
                ],
                uacs=[
                    (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["LSPV", "inner"],
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["LSPV", "inner", "anterior_posterior"],
                ],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                ],
            ),
        },
        "outer": {
            "segment_1": UACConfig(
                path=["LSPV", "outer"],
                markers=[
                    ["LSPV", "outer", "anterior_posterior"],
                    ["LSPV", "outer", "septal_lateral"],
                ],
                uacs=[
                    (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
                    (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["LSPV", "outer"],
                markers=[
                    ["LSPV", "outer", "septal_lateral"],
                    ["LSPV", "outer", "anchor"],
                ],
                uacs=[
                    (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
                    (
                        LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["LSPV", "outer"],
                markers=[
                    ["LSPV", "outer", "anchor"],
                    ["LSPV", "outer", "anterior_posterior"],
                ],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
                ],
            ),
        },
        "anterior_posterior": UACConfig(
            path=["LSPV", "anterior_posterior"],
            markers=[
                ["LSPV", "inner", "anterior_posterior"],
                ["LSPV", "outer", "anterior_posterior"],
            ],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["LSPV", "septal_lateral"],
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["LSPV", "outer", "septal_lateral"],
            ],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["LSPV", "anchor"],
            markers=[
                ["LSPV", "inner", "anchor"],
                ["LSPV", "outer", "anchor"],
            ],
            uacs=[
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "RSPV": {
        "inner": {
            "segment_1": UACConfig(
                path=["RSPV", "inner"],
                markers=[
                    ["RSPV", "inner", "anterior_posterior"],
                    ["RSPV", "inner", "septal_lateral"],
                ],
                uacs=[
                    (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                    (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["RSPV", "inner"],
                markers=[
                    ["RSPV", "inner", "septal_lateral"],
                    ["RSPV", "inner", "anchor"],
                ],
                uacs=[
                    (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["RSPV", "inner"],
                markers=[
                    ["RSPV", "inner", "anchor"],
                    ["RSPV", "inner", "anterior_posterior"],
                ],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                ],
            ),
        },
        "outer": {
            "segment_1": UACConfig(
                path=["RSPV", "outer"],
                markers=[
                    ["RSPV", "outer", "anterior_posterior"],
                    ["RSPV", "outer", "septal_lateral"],
                ],
                uacs=[
                    (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
                    (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["RSPV", "outer"],
                markers=[
                    ["RSPV", "outer", "septal_lateral"],
                    ["RSPV", "outer", "anchor"],
                ],
                uacs=[
                    (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
                    (
                        RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["RSPV", "outer"],
                markers=[
                    ["RSPV", "outer", "anchor"],
                    ["RSPV", "outer", "anterior_posterior"],
                ],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                    (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
                ],
            ),
        },
        "anterior_posterior": UACConfig(
            path=["RSPV", "anterior_posterior"],
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RSPV", "outer", "anterior_posterior"],
            ],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["RSPV", "septal_lateral"],
            markers=[
                ["RSPV", "inner", "septal_lateral"],
                ["RSPV", "outer", "septal_lateral"],
            ],
            uacs=[
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["RSPV", "anchor"],
            markers=[
                ["RSPV", "inner", "anchor"],
                ["RSPV", "outer", "anchor"],
            ],
            uacs=[
                (
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "RIPV": {
        "inner": {
            "segment_1": UACConfig(
                path=["RIPV", "inner"],
                markers=[
                    ["RIPV", "inner", "anterior_posterior"],
                    ["RIPV", "inner", "septal_lateral"],
                ],
                uacs=[
                    (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                    (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["RIPV", "inner"],
                markers=[
                    ["RIPV", "inner", "septal_lateral"],
                    ["RIPV", "inner", "anchor"],
                ],
                uacs=[
                    (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                    (
                        RIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["RIPV", "inner"],
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["RIPV", "inner", "anterior_posterior"],
                ],
                uacs=[
                    (
                        RIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                ],
            ),
        },
        "outer": {
            "segment_1": UACConfig(
                path=["RIPV", "outer"],
                markers=[
                    ["RIPV", "outer", "anterior_posterior"],
                    ["RIPV", "outer", "septal_lateral"],
                ],
                uacs=[
                    (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
                    (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
                ],
            ),
            "segment_2": UACConfig(
                path=["RIPV", "outer"],
                markers=[
                    ["RIPV", "outer", "septal_lateral"],
                    ["RIPV", "outer", "anchor"],
                ],
                uacs=[
                    (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
                    (
                        RIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "segment_3": UACConfig(
                path=["RIPV", "outer"],
                markers=[
                    ["RIPV", "outer", "anchor"],
                    ["RIPV", "outer", "anterior_posterior"],
                ],
                uacs=[
                    (
                        RIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                    ),
                    (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
                ],
            ),
        },
        "anterior_posterior": UACConfig(
            path=["RIPV", "anterior_posterior"],
            markers=[
                ["RIPV", "inner", "anterior_posterior"],
                ["RIPV", "outer", "anterior_posterior"],
            ],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["RIPV", "septal_lateral"],
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["RIPV", "outer", "septal_lateral"],
            ],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["RIPV", "anchor"],
            markers=[
                ["RIPV", "inner", "anchor"],
                ["RIPV", "outer", "anchor"],
            ],
            uacs=[
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "LAA": {
        "LAA": UACConfig(
            path=["LAA"],
            markers=[
                ["LAA", "LIPV"],
                ["LAA", "lateral"],
                ["LAA", "MV"],
                ["LAA", "posterior"],
            ],
            uacs=[
                (LAA_CENTER[0] - LAA_RADIUS, LAA_CENTER[1]),
                (LAA_CENTER[0], LAA_CENTER[1] - LAA_RADIUS),
                (LAA_CENTER[0] + LAA_RADIUS, LAA_CENTER[1]),
                (LAA_CENTER[0], LAA_CENTER[1] + LAA_RADIUS),
            ],
        ),
        "lateral": UACConfig(
            path=["LAA"],
            markers=[
                ["LAA", "LIPV"],
                ["LAA", "lateral"],
                ["LAA", "MV"],
            ],
            uacs=[
                (LIPV_CENTER[0] + LAA_DISTANCE, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                (
                    LIPV_CENTER[0] + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (LIPV_CENTER[0] + 3 * LAA_DISTANCE, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
        "posterior": UACConfig(
            path=["LAA"],
            markers=[
                ["LAA", "LIPV"],
                ["LAA", "posterior"],
                ["LAA", "MV"],
            ],
            uacs=[
                (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + LAA_DISTANCE),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DEPTH,
                    LIPV_CENTER[1] + 2 * LAA_DISTANCE,
                ),
                (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + 3 * LAA_DISTANCE),
            ],
        ),
    },
    "MV": {
        "anterior": UACConfig(
            path=["MV"],
            markers=[
                ["MV", "RSPV"],
                ["MV", "LSPV"],
            ],
            uacs=[
                (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
            ],
        ),
        "lateral": UACConfig(
            path=["MV"],
            markers=[
                ["MV", "LSPV"],
                ["MV", "LAA"],
            ],
            uacs=[
                (LSPV_CENTER[0] + ANCHOR_LENGTH, LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                (LIPV_CENTER[0] + ANCHOR_LENGTH, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
        "posterior": UACConfig(
            path=["MV"],
            markers=[
                ["MV", "LAA"],
                ["MV", "RIPV"],
            ],
            uacs=[
                (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + ANCHOR_LENGTH),
                (RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + ANCHOR_LENGTH),
            ],
        ),
        "septal": UACConfig(
            path=["MV"],
            markers=[
                ["MV", "RIPV"],
                ["MV", "RSPV"],
            ],
            uacs=[
                (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
    },
    "roof": {
        "LIPV_LSPV": UACConfig(
            path=["roof", "LIPV_LSPV"],
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LSPV", "inner", "anterior_posterior"],
            ],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        "LSPV_RSPV": UACConfig(
            path=["roof", "LSPV_RSPV"],
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["RSPV", "inner", "septal_lateral"],
            ],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        "RSPV_RIPV": UACConfig(
            path=["roof", "RSPV_RIPV"],
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RIPV", "inner", "anterior_posterior"],
            ],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        "RIPV_LIPV": UACConfig(
            path=["roof", "RIPV_LIPV"],
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["LIPV", "inner", "septal_lateral"],
            ],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
    },
    "anchor": {
        "LIPV_LAA": {
            "lateral": UACConfig(
                path=["anchor", "LIPV_LAA"],
                markers=[
                    ["LIPV", "inner", "anchor"],
                    ["LAA", "LIPV"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LIPV_CENTER[0] + LAA_DISTANCE, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
            "posterior": UACConfig(
                path=["anchor", "LIPV_LAA"],
                markers=[
                    ["LIPV", "inner", "anchor"],
                    ["LAA", "LIPV"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + LAA_DISTANCE),
                ],
            ),
        },
        "LAA_MV": {
            "lateral": UACConfig(
                path=["anchor", "LAA_MV"],
                markers=[
                    ["LAA", "MV"],
                    ["MV", "LAA"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + 3 * LAA_DISTANCE,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + ANCHOR_LENGTH,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "posterior": UACConfig(
                path=["anchor", "LAA_MV"],
                markers=[
                    ["LAA", "MV"],
                    ["MV", "LAA"],
                ],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + 3 * LAA_DISTANCE,
                    ),
                    (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + ANCHOR_LENGTH),
                ],
            ),
        },
        "LSPV_MV": {
            "anterior": UACConfig(
                path=["anchor", "LSPV_MV"],
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["MV", "LSPV"],
                ],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            "lateral_roof": UACConfig(
                path=["anchor", "LSPV_MV"],
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["MV", "LSPV"],
                ],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0] + ANCHOR_LENGTH, LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
            "lateral_mv": UACConfig(
                path=["anchor", "LSPV_MV"],
                markers=[
                    ["LSPV", "inner", "anchor"],
                    ["MV", "LSPV"],
                ],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0] + ANCHOR_LENGTH, 0),
                ],
            ),
        },
        "RSPV_MV": {
            "anterior": UACConfig(
                path=["anchor", "RSPV_MV"],
                markers=[
                    ["RSPV", "inner", "anchor"],
                    ["MV", "RSPV"],
                ],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            "septal": UACConfig(
                path=["anchor", "RSPV_MV"],
                markers=[
                    ["RSPV", "inner", "anchor"],
                    ["MV", "RSPV"],
                ],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
        },
        "RIPV_MV": {
            "septal": UACConfig(
                path=["anchor", "RIPV_MV"],
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["MV", "RIPV"],
                ],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
            "posterior_roof": UACConfig(
                path=["anchor", "RIPV_MV"],
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["MV", "RIPV"],
                ],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + ANCHOR_LENGTH),
                ],
            ),
            "posterior_mv": UACConfig(
                path=["anchor", "RIPV_MV"],
                markers=[
                    ["RIPV", "inner", "anchor"],
                    ["MV", "RIPV"],
                ],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RIPV_CENTER[1] + ANCHOR_LENGTH),
                ],
            ),
        },
        "LAA_lateral": UACConfig(
            path=["anchor", "LAA_lateral"],
            markers=[
                ["LAA", "lateral"],
                ["anchor", "LSPV_MV"],
            ],
            uacs=[
                (
                    LIPV_CENTER[0] + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (LSPV_CENTER[0] + ANCHOR_LENGTH, LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
        "LAA_posterior": UACConfig(
            path=["anchor", "LAA_posterior"],
            markers=[
                ["LAA", "posterior"],
                ["anchor", "RIPV_MV"],
            ],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DEPTH,
                    LIPV_CENTER[1] + 2 * LAA_DISTANCE,
                ),
                (RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + ANCHOR_LENGTH),
            ],
        ),
    },
}


# ==================================================================================================
submesh_configs = {
    "roof": SubmeshConfig(
        boundary_paths=[
            ["LIPV", "inner", "segment_1"],
            ["LSPV", "inner", "segment_1"],
            ["RSPV", "inner", "segment_1"],
            ["RIPV", "inner", "segment_1"],
            ["roof", "LIPV_LSPV"],
            ["roof", "LSPV_RSPV"],
            ["roof", "RSPV_RIPV"],
            ["roof", "RIPV_LIPV"],
        ],
        outside_path=["MV"],
    ),
    "anterior": SubmeshConfig(
        boundary_paths=[
            ["roof", "LSPV_RSPV"],
            ["anchor", "LSPV_MV", "anterior"],
            ["anchor", "RSPV_MV", "anterior"],
            ["LSPV", "inner", "segment_2"],
            ["RSPV", "inner", "segment_2"],
            ["MV", "anterior"],
        ],
        outside_path=["LIPV", "inner"],
    ),
    "septal": SubmeshConfig(
        boundary_paths=[
            ["roof", "RSPV_RIPV"],
            ["anchor", "RIPV_MV", "septal"],
            ["anchor", "RSPV_MV", "septal"],
            ["RIPV", "inner", "segment_3"],
            ["RSPV", "inner", "segment_3"],
            ["MV", "septal"],
        ],
        outside_path=["LIPV", "inner"],
    ),
    "posterior_roof": SubmeshConfig(
        boundary_paths=[
            ["roof", "RIPV_LIPV"],
            ["anchor", "RIPV_MV", "posterior_roof"],
            ["anchor", "LIPV_LAA", "posterior"],
            ["anchor", "LAA_posterior"],
            ["RIPV", "inner", "segment_2"],
            ["LIPV", "inner", "segment_2"],
            ["LAA", "posterior"],
        ],
        outside_path=["LSPV", "inner"],
    ),
    "posterior_mv": SubmeshConfig(
        boundary_paths=[
            ["anchor", "RIPV_MV", "posterior_mv"],
            ["anchor", "LAA_MV", "posterior"],
            ["anchor", "LAA_posterior"],
            ["LAA", "posterior"],
            ["MV", "posterior"],
        ],
        outside_path=["LSPV", "inner"],
    ),
    "lateral_roof": SubmeshConfig(
        boundary_paths=[
            ["roof", "LIPV_LSPV"],
            ["anchor", "LIPV_LAA", "lateral"],
            ["anchor", "LSPV_MV", "lateral_roof"],
            ["anchor", "LAA_lateral"],
            ["LIPV", "inner", "segment_3"],
            ["LSPV", "inner", "segment_3"],
            ["LAA"],
        ],
        outside_path=["RIPV", "inner"],
    ),
    "lateral_mv": SubmeshConfig(
        boundary_paths=[
            ["anchor", "LAA_MV", "lateral"],
            ["anchor", "LSPV_MV", "lateral_mv"],
            ["anchor", "LAA_lateral"],
            ["LAA", "lateral"],
            ["MV", "lateral"],
        ],
        outside_path=["RIPV", "inner"],
    ),
    "laa": SubmeshConfig(
        boundary_paths=[["LAA", "LAA"]],
        outside_path=["RIPV", "inner"],
    ),
}
