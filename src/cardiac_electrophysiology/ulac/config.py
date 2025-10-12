from dataclasses import dataclass

import numpy as np

# ruff: noqa: E501


# ==================================================================================================
@dataclass
class PathConfig:
    type: str
    name: tuple[str, ...]


@dataclass
class PointConfig:
    containing_path: str
    index: int


@dataclass
class BoundaryPathConfig:
    feature_tag: str
    coincides_with_mesh_boundary: bool


@dataclass
class ConnectionPathConfig:
    type: str
    start: str | PointConfig
    end: str | PointConfig


@dataclass
class MarkerConfig:
    point: PointConfig
    uacs: tuple[int, int]


@dataclass
class ParameterizationConfig:
    markers: tuple[str]
    marker_values: tuple[float]


@dataclass
class PatchBoundaryConfig:
    paths: tuple[str]
    portions: tuple[float, float]


# ==================================================================================================
boundary_path_configs = {
    "PV": {
        "LIPV": {
            "inner": BoundaryPathConfig("LIPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathConfig("LIPV", coincides_with_mesh_boundary=True)
            },
        "LSPV": {
            "inner": BoundaryPathConfig("LSPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathConfig("LSPV", coincides_with_mesh_boundary=True)
        },
        "RSPV": {
            "inner": BoundaryPathConfig("RSPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathConfig("RSPV", coincides_with_mesh_boundary=True)
        },
        "RIPV": {
            "inner": BoundaryPathConfig("RIPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathConfig("RIPV", coincides_with_mesh_boundary=True)
        },
    },
    "LAA": BoundaryPathConfig("LAA", coincides_with_mesh_boundary=False),
    "MV": BoundaryPathConfig("MV", coincides_with_mesh_boundary=True),
}


# ==================================================================================================
direct_connection_path_configs = {
    "roof": {
        "LIPV_LSPV": ConnectionPathConfig(start=("PV", "LIPV", "inner"), end=("PV", "LSPV", "inner")),
        "LSPV_RSPV": ConnectionPathConfig(start=("PV", "LSPV", "inner"), end=("PV", "RSPV", "inner")),
        "RSPV_RIPV": ConnectionPathConfig(start=("PV", "RSPV", "inner"), end=("PV", "RIPV", "inner")),
        "RIPV_LIPV": ConnectionPathConfig(start=("PV", "RIPV", "inner"), end=("PV", "LIPV", "inner")),
    },
    "diagonal": {
        "LAA_MV": ConnectionPathConfig(start=("LAA",), end=("MV",)),
        "LIPV_LAA": ConnectionPathConfig(start=("PV", "LIPV", "inner"), end=("LAA",)),
        "LSPV_MV": ConnectionPathConfig(start=("PV", "LSPV", "inner"), end=("MV",)),
        "RIPV_MV": ConnectionPathConfig(start=("PV", "RIPV", "inner"), end=("MV",)),
        "RSPV_MV": ConnectionPathConfig(start=("PV", "RSPV", "inner"), end=("MV",)),
    },
}


# ==================================================================================================
indirect_connection_path_configs = {
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "LIPV_LSPV"), 0), end=("PV", "LIPV", "outer")),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "RIPV_LIPV"), -1), end=("PV", "LIPV", "outer")),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "LIPV_LAA"), 0), end=("PV", "LIPV", "outer")),
        },
        "LSPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "LIPV_LSPV"), -1), end=("PV", "LSPV", "inner")),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "RIPV_LIPV"), -1), end=("PV", "LSPV", "inner")),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "LSPV_MV"), 0), end=("PV", "LSPV", "inner")),
        },
        "RSPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "RSPV_RIPV"), 0), end=("PV", "RSPV", "inner")),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "LSPV_RSPV"), -1), end=("PV", "RSPV", "inner")),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "RSPV_MV"), 0), end=("PV", "RSPV", "inner")),
        },
        "RIPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "RSPV_RIPV"), index=-1), end=("PV", "RIPV", "inner")),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "RIPV_LIPV"), index=0), end=("PV", "RIPV", "inner")),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "RIPV_MV"), index=0), end=("PV", "RIPV", "inner")),
        },
    },
    "laa_segments": {
        "posterior": ConnectionPathConfig(start="LAA", end=("diagonal", "RIPV_MV")),
        "lateral": ConnectionPathConfig(start=("LAA"), end=("diagonal", "LSPV_MV")),
    }
}


# ==================================================================================================
PV_INNER_RADIUS = 0.08
PV_OUTER_RADIUS = 0.04
LAA_RADIUS = 0.08
LIPV_CENTER = (2 / 3, 2 / 3)
LSPV_CENTER = (2 / 3, 1 / 3)
RIPV_CENTER = (1 / 3, 2 / 3)
RSPV_CENTER = (1 / 3, 1 / 3)
LAA_CENTER = (5 / 6, 5 / 6)

marker_configs = {
    # ---------- PVs ----------
    "PV": {
        "LIPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "lipv", "anterior_posterior"), 0),
                                                   uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "lipv", "septal_lateral"), 0),
                                               uacs=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "lipv", "diagonal"), 0),
                                         uacs=(LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "lipv", "anterior_posterior"), -1),
                                                   uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "lipv", "septal_lateral"), -1),
                                               uacs=(LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "lipv", "diagonal"), -1),
                                         uacs=(LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "LSPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "lspv", "anterior_posterior"), 0),
                                                   uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "lspv", "septal_lateral"), 0),
                                               uacs=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "lspv", "diagonal"), 0),
                                         uacs=(LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "lspv", "anterior_posterior"), -1),
                                                   uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "lspv", "septal_lateral"), -1),
                                               uacs=(LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "lspv", "diagonal"), -1),
                                         uacs=(LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RSPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "rspv", "anterior_posterior"), 0),
                                                   uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "rspv", "septal_lateral"), 0),
                                               uacs=(RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "rspv", "diagonal"), 0),
                                         uacs=(RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "rspv", "anterior_posterior"), -1),
                                                   uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "rspv", "septal_lateral"), -1),
                                               uacs=(RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "rspv", "diagonal"), -1),
                                         uacs=(RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RIPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "ripv", "anterior_posterior"), 0),
                                                   uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "ripv", "septal_lateral"), 0),
                                               uacs=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "ripv", "diagonal"), 0),
                                         uacs=(RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "ripv", "anterior_posterior"), -1),
                                                   uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "ripv", "septal_lateral"), -1),
                                               uacs=(RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "ripv", "diagonal"), -1),
                                         uacs=(RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
    },

    # ---------- LAA ----------
    "LAA": {
        "LIPV": MarkerConfig(PointConfig(("diagonal", "LIPV_LAA"), -1),
                             uacs=((LAA_CENTER[0] - LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] - LAA_RADIUS) / np.sqrt(2))),
        "MV": MarkerConfig(PointConfig(("diagonal", "LIPV_LAA"), 0),
                           uacs=((LAA_CENTER[0] + LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] + LAA_RADIUS) / np.sqrt(2))),
        "posterior": MarkerConfig(PointConfig(("laa_segments", "posterior"), 0),
                                  uacs=((LAA_CENTER[0] - LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] + LAA_RADIUS) / np.sqrt(2))),
        "lateral": MarkerConfig(PointConfig(("laa_segments", "lateral"), 0),
                               uacs=((LAA_CENTER[0] + LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] - LAA_RADIUS) / np.sqrt(2))),
    },

    # ---------- MV ----------
    "MV": {
        "LAA": MarkerConfig(PointConfig(("diagonal","LAA_MV"), -1), uacs=(1, 1)),
        "LSPV": MarkerConfig(PointConfig(("diagonal", "LSPV_MV"), -1), uacs=(1, 0)),
        "RSPV": MarkerConfig(PointConfig(("diagonal", "RSPV_MV"), -1), uacs=(0, 0)),
        "RIPV": MarkerConfig(PointConfig(("diagonal", "RIPV_MV"), -1), uacs=(0, 1)),
    },
    "RIPV_MV": MarkerConfig(PointConfig(("laa_segments", "posterior"), -1),
                            uacs=(RIPV_CENTER[0] / 2, (1 + RIPV_CENTER[1]) / 2)),
    "LSPV_MV": MarkerConfig(PointConfig(("laa_segments", "posterior"), -1),
                            uacs=((1 + LSPV_CENTER[0]) / 2, LSPV_CENTER[1] / 2)),
}


# ==================================================================================================
parameterization_configs = {
    # ---------- PV boundaries ----------
    "pv_boundaries": {
        "LIPV": {
            "inner": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                     ("PV", "LIPV", "inner", "septal_lateral"),
                                                     ("PV", "LIPV", "inner", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationConfig(markers=(("PV", "LIPV", "outer", "anterior_posterior"),
                                                     ("PV", "LIPV", "outer", "septal_lateral"),
                                                     ("PV", "LIPV", "outer", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            },
        "LSPV": {
            "inner": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                     ("PV", "LSPV", "inner", "septal_lateral"),
                                                     ("PV", "LSPV", "inner", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationConfig(markers=(("PV", "LSPV", "outer", "anterior_posterior"),
                                                     ("PV", "LSPV", "outer", "septal_lateral"),
                                                     ("PV", "LSPV", "outer", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            },
        "RSPV": {
            "inner": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                     ("PV", "RSPV", "inner", "septal_lateral"),
                                                     ("PV", "RSPV", "inner", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationConfig(markers=(("PV", "RSPV", "outer", "anterior_posterior"),
                                                     ("PV", "RSPV", "outer", "septal_lateral"),
                                                     ("PV", "RSPV", "outer", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            },
        "RIPV": {
            "inner": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                     ("PV", "RIPV", "inner", "septal_lateral"),
                                                     ("PV", "RIPV", "inner", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationConfig(markers=(("PV", "RIPV", "outer", "anterior_posterior"),
                                                     ("PV", "RIPV", "outer", "septal_lateral"),
                                                     ("PV", "RIPV", "outer", "diagonal")),
                                            marker_values=(0, 1 / 3, 2 / 3)),
            },
    },

    # ---------- PV segments ----------
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                                  ("PV", "LIPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "septal_lateral"),
                                                              ("PV", "LIPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "diagonal"),
                                                        ("PV", "LIPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "LSPV": {
            "anterior_posterior": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                                  ("PV", "LSPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                              ("PV", "LSPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "diagonal"),
                                                        ("PV", "LSPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "RSPV": {
            "anterior_posterior": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                                  ("PV", "RSPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "septal_lateral"),
                                                              ("PV", "RSPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "diagonal"),
                                                        ("PV", "RSPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "RIPV": {
            "anterior_posterior": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                                  ("PV", "RIPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                              ("PV", "RIPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "diagonal"),
                                                        ("PV", "RIPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
    },

    # ---------- Roof ----------
    "roof": {
        "LIPV_LSPV": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                     ("PV", "LSPV", "inner", "anterior_posterior")),
                                            marker_values=(0, 1)),
        "LSPV_RSPV": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                     ("PV", "RSPV", "inner", "septal_lateral")),
                                            marker_values=(0, 1)),
        "RSPV_RIPV": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                     ("PV", "RIPV", "inner", "anterior_posterior")),
                                            marker_values=(0, 1)),
        "RIPV_LIPV": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                     ("PV", "LIPV", "inner", "septal_lateral")),
                                            marker_values=(0, 1)),
    },

    # ---------- Diagonals ----------
    "diagonal": {
        "LIPV_LAA": ParameterizationConfig(markers=(("PV", "LIPV", "inner", "diagonal"), ("LAA", "LIPV")),
                                           marker_values=(0, 1)),
        "LAA_MV": ParameterizationConfig(markers=(("LAA", "LIPV"), ("MV", "LAA")),
                                         marker_values=(0, 1)),
        "LSPV_MV": ParameterizationConfig(markers=(("PV", "LSPV", "inner", "diagonal"), ("LSPV_MV",), ("MV", "LSPV")),
                                          marker_values=(0, 1 / 2, 1)),
        "RSPV_MV": ParameterizationConfig(markers=(("PV", "RSPV", "inner", "diagonal"), ("MV", "RSPV")),
                                          marker_values=(0, 1)),
        "RIPV_MV": ParameterizationConfig(markers=(("PV", "RIPV", "inner", "diagonal"), ("LSPV_MV",), ("MV", "RIPV")),
                                          marker_values=(0, 1 / 2, 1)),
    },

    # ---------- LAA Segments ----------
    "laa_segments": {
        "posterior": ParameterizationConfig(markers=(("LAA", "posterior"), ("RIPV_MV",)),
                                            marker_values=(0, 1)),
        "lateral": ParameterizationConfig(markers=(("LAA", "lateral"), ("LSPV_MV",)),
                                          marker_values=(0, 1)),
    },

    # ---------- LAA and PV ----------
    "LAA": ParameterizationConfig(markers=(("LAA", "LIPV"), ("LAA", "lateral"), ("LAA", "MV"), ("LAA", "posterior")),
                                  marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
    "MV": ParameterizationConfig(markers=(("MV", "RSPV"), ("MV", "LSPV"), ("MV", "LAA"), ("MV", "RIPV")),
                                 marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
}


# ==================================================================================================
patch_boundary_configs = {
    "roof": PatchBoundaryConfig(paths=(("roof", "LIPV_LSPV"), ("roof", "LSPV_RSPV"),
                                       ("roof", "RSPV_RIPV"), ("roof", "RIPV_LIPV"),
                                       ("pv_boundaries", "LIPV", "inner"), ("pv_boundaries", "LSPV", "inner"),
                                       ("pv_boundaries", "RSPV", "inner"), ("pv_boundaries", "RIPV", "inner")),
                                portions=((0, 1), (0, 1),
                                          (0, 1), (0, 1),
                                          (0, 1 / 4), (0, 1 / 4),
                                          (0, 1 / 4), (0, 1 / 4))),

    "anterior": PatchBoundaryConfig(paths=(("roof", "LSPV_RSPV"),
                                           ("diagonal", "LSPV_MV"), ("diagonal", "RSPV_MV"),
                                           ("pv_boundaries", "LSPV", "inner"), ("pv_boundaries", "RSPV", "inner"),
                                           ("mv_boundary",)),
                                    portions=((0, 1),
                                              (0, 1), (0, 1),
                                              (1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                              (0, 1 / 4))),
    "septal": PatchBoundaryConfig(paths=(("roof", "RSPV_RIPV"),
                                         ("diagonal", "RIPV_MV"), ("diagonal", "RSPV_MV"),
                                         ("pv_boundaries", "RIPV", "inner"), ("pv_boundaries", "RSPV", "inner"),
                                         ("mv_boundary",)),
                                  portions=((0, 1),
                                            (0, 1), (0, 1),
                                            (5 / 8, 1), (5 / 8, 1),
                                            (3 / 4, 1))),
    "posterior_roof": PatchBoundaryConfig(paths=(("roof", "RIPV_LIPV"),
                                                 ("diagonal", "RIPV_MV"), ("diagonal", "LIPV_LAA"),
                                                 ("pv_boundaries", "RIPV", "inner"), ("pv_boundaries", "LIPV", "inner"),
                                                 ("laa_segments", "posterior"), ("laa_boundary",),
                                                 ("mv_boundary",)),
                                    portions=((0, 1),
                                              (1 / 2, 1), (0, 1),
                                              (1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                              (0, 1),
                                              (3 / 4, 1),
                                              (1 / 2, 3 / 4))),
    "posterior_mv": PatchBoundaryConfig(paths=(("diagonal", "RIPV_MV"), ("diagonal", "LAA_MV"),
                                               ("laa_segments", "posterior"), ("laa_boundary",),
                                               ("mv_boundary",)),
                                        portions=((1 / 2, 1), (0, 1),
                                                  (0, 1),
                                                  (1 / 2, 3 / 4),
                                                  (1 / 2, 3 / 4))),
    "lateral_roof": PatchBoundaryConfig(paths=(("roof", "LIPV_LSPV"),
                                               ("diagonal", "LIPV_LAA"), ("diagonal", "LSPV_MV"),
                                               ("pv_boundaries", "LIPV", "inner"), ("pv_boundaries", "LSPV", "inner"),
                                               ("laa_segments", "lateral"), ("laa_boundary",)),
                                        portions=((0, 1),
                                                  (0, 1), (0, 1 / 2),
                                                  (5 / 8, 1), (5 / 8, 1),
                                                  (0, 1),
                                                  (0, 1 / 4))),
    "lateral_mv": PatchBoundaryConfig(paths=(("diagonal", "LAA_MV"), ("diagonal", "LSPV_MV"),
                                             ("laa_segments", "lateral"), ("laa_boundary",)),
                                      portions=((0, 1), (1 / 2, 1),
                                                (0, 1),
                                                (1 / 4, 1 / 2))),
    "pv_segments": {
        "LIPV": {
            "segment_1": PatchBoundaryConfig(paths=(("pv_boundaries", "LIPV", "inner"), ("pv_boundaries", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "anterior_posterior"), ("pv_segments", "LIPV", "septal_lateral")),
                                             portions=((0, 1 / 4), (0, 1 / 4),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("pv_boundaries", "LIPV", "inner"), ("pv_boundaries", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "septal_lateral"), ("pv_segments", "LIPV", "diagonal")),
                                             portions=((1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("pv_boundaries", "LIPV", "inner"), ("pv_boundaries", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "diagonal"), ("pv_segments", "LIPV", "anterior_posterior")),
                                             portions=((5 / 8, 1), (5 / 8, 1),
                                                       (0, 1), (0, 1))),
        },
        "LSPV": {
            "segment_1": PatchBoundaryConfig(paths=(("pv_boundaries", "LSPV", "inner"), ("pv_boundaries", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "anterior_posterior"), ("pv_segments", "LSPV", "septal_lateral")),
                                             portions=((0, 1 / 4), (0, 1 / 4),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("pv_boundaries", "LSPV", "inner"), ("pv_boundaries", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "septal_lateral"), ("pv_segments", "LSPV", "diagonal")),
                                             portions=((1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("pv_boundaries", "LSPV", "inner"), ("pv_boundaries", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "diagonal"), ("pv_segments", "LSPV", "anterior_posterior")),
                                             portions=((5 / 8, 1), (5 / 8, 1),
                                                       (0, 1), (0, 1))),
        },
        "RSPV": {
            "segment_1": PatchBoundaryConfig(paths=(("pv_boundaries", "RSPV", "inner"), ("pv_boundaries", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "anterior_posterior"), ("pv_segments", "RSPV", "septal_lateral")),
                                             portions=((0, 1 / 4), (0, 1 / 4),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("pv_boundaries", "RSPV", "inner"), ("pv_boundaries", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "septal_lateral"), ("pv_segments", "RSPV", "diagonal")),
                                             portions=((1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("pv_boundaries", "RSPV", "inner"), ("pv_boundaries", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "diagonal"), ("pv_segments", "RSPV", "anterior_posterior")),
                                             portions=((5 / 8, 1), (5 / 8, 1),
                                                       (0, 1), (0, 1))),
        },
        "RIPV": {
            "segment_1": PatchBoundaryConfig(paths=(("pv_boundaries", "RIPV", "inner"), ("pv_boundaries", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "anterior_posterior"), ("pv_segments", "RIPV", "septal_lateral")),
                                             portions=((0, 1 / 4), (0, 1 / 4),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("pv_boundaries", "RIPV", "inner"), ("pv_boundaries", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "septal_lateral"), ("pv_segments", "RIPV", "diagonal")),
                                             portions=((1 / 4, 5 / 8), (1 / 4, 5 / 8),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("pv_boundaries", "RIPV", "inner"), ("pv_boundaries", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "diagonal"), ("pv_segments", "RIPV", "anterior_posterior")),
                                             portions=((5 / 8, 1), (5 / 8, 1),
                                                       (0, 1), (0, 1))),
        },
    },
}
