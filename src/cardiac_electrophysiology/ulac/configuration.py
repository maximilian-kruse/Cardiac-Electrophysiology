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
    uacs: list[float, float] | tuple[float, float]


@dataclass
class ParameterizationConfig:
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
class SubmeshConfig:
    boundary_paths: Iterable[Iterable[str]]
    portions: Iterable[list[float, float] | tuple[float, float]]
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
PV_SEPTAL_LATERAL = 1 / 4
PV_ANCHOR = 5 / 8

parameterization_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LIPV", "inner", "septal_lateral"],
                ["LIPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            markers=[
                ["LIPV", "outer", "anterior_posterior"],
                ["LIPV", "outer", "septal_lateral"],
                ["LIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1 / 4, 5 / 8],
        ),
        "anterior_posterior": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LIPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "septal_lateral"],
                ["LIPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "anchor"],
                ["LIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LSPV": {
        "inner": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "anterior_posterior"],
                ["LSPV", "inner", "septal_lateral"],
                ["LSPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            markers=[
                ["LSPV", "outer", "anterior_posterior"],
                ["LSPV", "outer", "septal_lateral"],
                ["LSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "anterior_posterior"],
                ["LSPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["LSPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "anchor"],
                ["LSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RSPV": {
        "inner": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RSPV", "inner", "septal_lateral"],
                ["RSPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            markers=[
                ["RSPV", "outer", "anterior_posterior"],
                ["RSPV", "outer", "septal_lateral"],
                ["RSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RSPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "septal_lateral"],
                ["RSPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "anchor"],
                ["RSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RIPV": {
        "inner": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "anterior_posterior"],
                ["RIPV", "inner", "septal_lateral"],
                ["RIPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            markers=[
                ["RIPV", "outer", "anterior_posterior"],
                ["RIPV", "outer", "septal_lateral"],
                ["RIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "anterior_posterior"],
                ["RIPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["RIPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "anchor"],
                ["RIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": {
        "LAA": ParameterizationConfig(
            markers=[
                ["LAA", "LIPV", "LAA"],
                ["LAA", "lateral", "LAA"],
                ["LAA", "MV", "LAA"],
                ["LAA", "posterior", "LAA"],
            ],
            marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
        ),
        "lateral": ParameterizationConfig(
            markers=[
                ["LAA", "LIPV", "lateral"],
                ["LAA", "lateral", "lateral"],
                ["LAA", "MV", "lateral"],
            ],
            marker_relative_positions=[0, 1 / 4, 1 / 2],
        ),
        "posterior": ParameterizationConfig(
            markers=[
                ["LAA", "LIPV", "posterior"],
                ["LAA", "posterior", "posterior"],
                ["LAA", "MV", "posterior"],
            ],
            marker_relative_positions=[0, 1 / 4, 1 / 2],
        ),
    },
    # ------------------------------------------------------------------------------------------------------
    "MV": {
        "anterior": ParameterizationConfig(
            markers=[
                ["MV", "RSPV", "anterior"],
                ["MV", "LSPV", "anterior"],
            ],
            marker_relative_positions=[]
        ),
    },
    "MV": ParameterizationConfig(
        markers=[
            ["MV", "RSPV"],
            ["MV", "LSPV"],
            ["MV", "LAA"],
            ["MV", "RIPV"],
        ],
        marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
    ),
    # roof
    "roof": {
        "LIPV_LSPV": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LSPV", "inner", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LSPV_RSPV": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["RSPV", "inner", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RSPV_RIPV": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RIPV", "inner", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_LIPV": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["LIPV", "inner", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # anchor
    "anchor": {
        "LIPV_LAA": ParameterizationConfig(
            markers=[
                ["LIPV", "inner", "anchor"],
                ["LAA", "LIPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LAA_MV": ParameterizationConfig(
            markers=[
                ["LAA", "MV"],
                ["MV", "LAA"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LSPV_MV": ParameterizationConfig(
            markers=[
                ["LSPV", "inner", "anchor"],
                ["MV", "LSPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RSPV_MV": ParameterizationConfig(
            markers=[
                ["RSPV", "inner", "anchor"],
                ["MV", "RSPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_MV": ParameterizationConfig(
            markers=[
                ["RIPV", "inner", "anchor"],
                ["MV", "RIPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LAA_lateral": ParameterizationConfig(
            markers=[
                ["LAA", "lateral"],
                ["anchor", "LSPV_MV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LAA_posterior": ParameterizationConfig(
            markers=[
                ["LAA", "posterior"],
                ["anchor", "RIPV_MV"],
            ],
            marker_relative_positions=[0, 1],
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
            position=0.9,
            uacs=(3 / 5 + 0.9 * 2 / 5, 0.1 * 1 / 3),
        ),
        "RIPV_MV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="relative",
            position=0.9,
            uacs=(0.1 * 2 / 5, 2 / 3 + 0.9 * 1 / 3),
        ),
    },
}


# ==================================================================================================
submesh_configs = {
    "roof": SubmeshConfig(
        boundary_paths=[
            ["LIPV", "inner"],
            ["LSPV", "inner"],
            ["RSPV", "inner"],
            ["RIPV", "inner"],
            ["roof", "LIPV_LSPV"],
            ["roof", "LSPV_RSPV"],
            ["roof", "RSPV_RIPV"],
            ["roof", "RIPV_LIPV"],
        ],
        portions=[(0, 1 / 4), (0, 1 / 4), (0, 1 / 4), (0, 1 / 4), (0, 1), (0, 1), (0, 1), (0, 1)],
        outside_path=["MV"],
    ),
    "anterior": SubmeshConfig(
        boundary_paths=[
            ["roof", "LSPV_RSPV"],
            ["anchor", "LSPV_MV"],
            ["anchor", "RSPV_MV"],
            ["LSPV", "inner"],
            ["RSPV", "inner"],
            ["MV"],
        ],
        portions=[
            (0, 1),
            (0, 1),
            (0, 1),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (0, 1 / 4),
        ],
        outside_path=["LIPV", "inner"],
    ),
    "septal": SubmeshConfig(
        boundary_paths=[
            ["roof", "RSPV_RIPV"],
            ["anchor", "RIPV_MV"],
            ["anchor", "RSPV_MV"],
            ["RIPV", "inner"],
            ["RSPV", "inner"],
            ["MV"],
        ],
        portions=[(0, 1), (0, 1), (0, 1), (PV_ANCHOR, 1), (PV_ANCHOR, 1), (3 / 4, 1)],
        outside_path=["LIPV", "inner"],
    ),
    "posterior_roof": SubmeshConfig(
        boundary_paths=[
            ["roof", "RIPV_LIPV"],
            ["anchor", "RIPV_MV"],
            ["anchor", "LIPV_LAA"],
            ["anchor", "LAA_posterior"],
            ["RIPV", "inner"],
            ["LIPV", "inner"],
            ["LAA"],
        ],
        portions=[
            (0, 1),
            (0, 0.9),
            (0, 1),
            (0, 1),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (3 / 4, 1),
        ],
        outside_path=["LSPV", "inner"],
    ),
    "posterior_mv": SubmeshConfig(
        boundary_paths=[
            ["anchor", "RIPV_MV"],
            ["anchor", "LAA_MV"],
            ["anchor", "LAA_posterior"],
            ["LAA"],
            ["MV"],
        ],
        portions=[(0.9, 1), (0, 1), (0, 1), (1 / 2, 3 / 4), (1 / 2, 3 / 4)],
        outside_path=["LSPV", "inner"],
    ),
    "lateral_roof": SubmeshConfig(
        boundary_paths=[
            ["roof", "LIPV_LSPV"],
            ["anchor", "LIPV_LAA"],
            ["anchor", "LSPV_MV"],
            ["anchor", "LAA_lateral"],
            ["LIPV", "inner"],
            ["LSPV", "inner"],
            ["LAA"],
        ],
        portions=[
            (0, 1),
            (0, 1),
            (0, 0.9),
            (0, 1),
            (PV_ANCHOR, 1),
            (PV_ANCHOR, 1),
            (0, 1 / 4),
        ],
        outside_path=["RIPV", "inner"],
    ),
    # "lateral_mv": SubmeshConfig(
    #     boundary_paths=[
    #         ["anchor", "LAA_MV"],
    #         ["anchor", "LSPV_MV"],
    #         ["anchor", "LAA_lateral"],
    #         ["LAA"],
    #         ["MV"],
    #     ],
    #     portions=[
    #         (0, 1),
    #         (0.9, 1),
    #         (0, 1),
    #         (1 / 4, 1 / 2),
    #         (1 / 4, 1 / 2),
    #     ],
    #     outside_path=["RIPV", "inner"],
    # ),
    "laa": SubmeshConfig(
        boundary_paths=[["LAA"]],
        portions=[(0, 1)],
        outside_path=["RIPV", "inner"],
    ),
}
