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
    start: str | PointConfig
    end: str | PointConfig
    inadmissible: tuple[tuple[str]]


@dataclass
class MarkerConfig:
    point: PointConfig
    uacs: tuple[int, int]


@dataclass
class ParameterizationConfig:
    path: tuple[str]
    markers: tuple[str]
    marker_values: tuple[float]


@dataclass
class PatchBoundaryConfig:
    paths: tuple[str]
    portions: tuple[float, float]


@dataclass
class PatchConfig:
    boundary: tuple[str]
    outside: str


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
        "LIPV_LSPV": ConnectionPathConfig(start=("PV", "LIPV", "inner"),
                                          end=("PV", "LSPV", "inner"),
                                          inadmissible=(("PV", "LIPV", "inner"), ("PV", "LSPV", "inner"))),
        "LSPV_RSPV": ConnectionPathConfig(start=("PV", "LSPV", "inner"),
                                          end=("PV", "RSPV", "inner"),
                                          inadmissible=(("PV", "LSPV", "inner"), ("PV", "RSPV", "inner"))),
        "RSPV_RIPV": ConnectionPathConfig(start=("PV", "RSPV", "inner"),
                                          end=("PV", "RIPV", "inner"),
                                          inadmissible=(("PV", "RSPV", "inner"), ("PV", "RIPV", "inner"))),
        "RIPV_LIPV": ConnectionPathConfig(start=("PV", "RIPV", "inner"), end=("PV", "LIPV", "inner"),
                                          inadmissible=(("PV", "RIPV", "inner"), ("PV", "LIPV", "inner"))),
    },
    "diagonal": {
        "LAA_MV": ConnectionPathConfig(start=("LAA",),
                                       end=("MV",),
                                       inadmissible=(("LAA",), ("MV",))),
        "LIPV_LAA": ConnectionPathConfig(start=("PV", "LIPV", "inner"),
                                         end=("LAA",),
                                         inadmissible=(("PV", "LIPV", "inner"), ("LAA",))),
        "LSPV_MV": ConnectionPathConfig(start=("PV", "LSPV", "inner"),
                                        end=("MV",),
                                        inadmissible=(("PV", "LSPV", "inner"), ("MV",))),
        "RIPV_MV": ConnectionPathConfig(start=("PV", "RIPV", "inner"),
                                        end=("MV",),
                                        inadmissible=(("PV", "RIPV", "inner"), ("MV",))),
        "RSPV_MV": ConnectionPathConfig(start=("PV", "RSPV", "inner"),
                                        end=("MV",),
                                        inadmissible=(("PV", "RSPV", "inner"), ("MV",))),
    },
}


# ==================================================================================================
indirect_connection_path_configs = {
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "LIPV_LSPV"), 0),
                                                       end=("PV", "LIPV", "outer"),
                                                       inadmissible=(("roof", "LIPV_LSPV"),
                                                                     ("PV", "LIPV", "inner"),
                                                                     ("PV", "LIPV", "outer"))),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "RIPV_LIPV"), -1),
                                                   end=("PV", "LIPV", "outer"),
                                                   inadmissible=(("roof", "RIPV_LIPV"),
                                                                 ("PV", "LIPV", "inner"),
                                                                 ("PV", "LIPV", "outer"))),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "LIPV_LAA"), 0),
                                             end=("PV", "LIPV", "outer"),
                                             inadmissible=(("diagonal", "LIPV_LAA"),
                                                           ("PV", "LIPV", "inner"),
                                                           ("PV", "LIPV", "outer"))),
        },
        "LSPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "LIPV_LSPV"), -1),
                                                       end=("PV", "LSPV", "outer"),
                                                       inadmissible=(("roof", "LIPV_LSPV"),
                                                                     ("PV", "LSPV", "inner"),
                                                                     ("PV", "LSPV", "outer"))),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "LSPV_RSPV"), 0),
                                                   end=("PV", "LSPV", "outer"),
                                                   inadmissible=(("roof", "LSPV_RSPV"),
                                                                 ("PV", "LSPV", "inner"),
                                                                 ("PV", "LSPV", "outer"))),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "LSPV_MV"), 0),
                                             end=("PV", "LSPV", "outer"),
                                             inadmissible=(("diagonal", "LSPV_MV"),
                                                            ("PV", "LSPV", "inner"),
                                                            ("PV", "LSPV", "outer"))),
        },
        "RSPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "RSPV_RIPV"), 0),
                                                       end=("PV", "RSPV", "outer"),
                                                       inadmissible=(("roof", "RSPV_RIPV"),
                                                                      ("PV", "RSPV", "inner"),
                                                                      ("PV", "RSPV", "outer"))),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "LSPV_RSPV"), -1),
                                                   end=("PV", "RSPV", "outer"),
                                                   inadmissible=(("roof", "LSPV_RSPV"),
                                                                  ("PV", "RSPV", "inner"),
                                                                  ("PV", "RSPV", "outer"))),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "RSPV_MV"), 0),
                                             end=("PV", "RSPV", "outer"),
                                             inadmissible=(("diagonal", "RSPV_MV"),
                                                            ("PV", "RSPV", "inner"),
                                                            ("PV", "RSPV", "outer"))),
        },
        "RIPV": {
            "anterior_posterior": ConnectionPathConfig(start=PointConfig(("roof", "RSPV_RIPV"), -1),
                                                       end=("PV", "RIPV", "outer"),
                                                       inadmissible=(("roof", "RSPV_RIPV"),
                                                                     ("PV", "RIPV", "inner"),
                                                                     ("PV", "RIPV", "outer"))),
            "septal_lateral": ConnectionPathConfig(start=PointConfig(("roof", "RIPV_LIPV"), 0),
                                                   end=("PV", "RIPV", "outer"),
                                                   inadmissible=(("roof", "RIPV_LIPV"),
                                                                 ("PV", "RIPV", "inner"),
                                                                 ("PV", "RIPV", "outer"))),
            "diagonal": ConnectionPathConfig(start=PointConfig(("diagonal", "RIPV_MV"), 0),
                                              end=("PV", "RIPV", "outer"),
                                              inadmissible=(("diagonal", "RIPV_MV"),
                                                            ("PV", "RIPV", "inner"),
                                                            ("PV", "RIPV", "outer"))),
        },
    },
    "laa_segments": {
        "posterior": ConnectionPathConfig(start=("LAA",),
                                          end=("diagonal", "RIPV_MV"),
                                          inadmissible=(("LAA",), ("MV",), ("diagonal", "RIPV_MV"))),
        "lateral": ConnectionPathConfig(start=("LAA",),
                                        end=("diagonal", "LSPV_MV"),
                                        inadmissible=(("LAA",), ("MV",), ("diagonal", "LSPV_MV"))),
    }
}


# ==================================================================================================
PV_INNER_RADIUS = 0.08
PV_OUTER_RADIUS = 0.04
LAA_RADIUS = 0.12
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
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "LIPV", "anterior_posterior"), 0),
                                                   uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "LIPV", "septal_lateral"), 0),
                                               uacs=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "LIPV", "diagonal"), 0),
                                         uacs=(LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "LIPV", "anterior_posterior"), -1),
                                                   uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "LIPV", "septal_lateral"), -1),
                                               uacs=(LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "LIPV", "diagonal"), -1),
                                         uacs=(LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "LSPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "LSPV", "anterior_posterior"), 0),
                                                   uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "LSPV", "septal_lateral"), 0),
                                               uacs=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "LSPV", "diagonal"), 0),
                                         uacs=(LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "LSPV", "anterior_posterior"), -1),
                                                   uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "LSPV", "septal_lateral"), -1),
                                               uacs=(LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "LSPV", "diagonal"), -1),
                                         uacs=(LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RSPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "RSPV", "anterior_posterior"), 0),
                                                   uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "RSPV", "septal_lateral"), 0),
                                               uacs=(RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "RSPV", "diagonal"), 0),
                                         uacs=(RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "RSPV", "anterior_posterior"), -1),
                                                   uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "RSPV", "septal_lateral"), -1),
                                               uacs=(RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "RSPV", "diagonal"), -1),
                                         uacs=(RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RIPV": {
            "inner": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "RIPV", "anterior_posterior"), 0),
                                                   uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "RIPV", "septal_lateral"), 0),
                                               uacs=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "RIPV", "diagonal"), 0),
                                         uacs=(RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerConfig(PointConfig(("pv_segments", "RIPV", "anterior_posterior"), -1),
                                                   uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerConfig(PointConfig(("pv_segments", "RIPV", "septal_lateral"), -1),
                                               uacs=(RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerConfig(PointConfig(("pv_segments", "RIPV", "diagonal"), -1),
                                         uacs=(RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
    },

    # ---------- LAA ----------
    "LAA": {
        "LIPV": MarkerConfig(PointConfig(("diagonal", "LIPV_LAA"), -1),
                             uacs=(LAA_CENTER[0] - LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2))),
        "MV": MarkerConfig(PointConfig(("diagonal", "LAA_MV"), 0),
                           uacs=(LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2))),
        "posterior": MarkerConfig(PointConfig(("laa_segments", "posterior"), 0),
                                  uacs=(LAA_CENTER[0] - LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2))),
        "lateral": MarkerConfig(PointConfig(("laa_segments", "lateral"), 0),
                               uacs=(LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2))),
    },

    # ---------- MV ----------
    "MV": {
        "LAA": MarkerConfig(PointConfig(("diagonal","LAA_MV"), -1), uacs=(1, 1)),
        "LSPV": MarkerConfig(PointConfig(("diagonal", "LSPV_MV"), -1), uacs=(1, 0)),
        "RSPV": MarkerConfig(PointConfig(("diagonal", "RSPV_MV"), -1), uacs=(0, 0)),
        "RIPV": MarkerConfig(PointConfig(("diagonal", "RIPV_MV"), -1), uacs=(0, 1)),
    },
    "RIPV_MV": MarkerConfig(PointConfig(("laa_segments", "posterior"), -1),
                            uacs=(1 - LAA_CENTER[0]- LAA_RADIUS / np.sqrt(2), LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2))),
    "LSPV_MV": MarkerConfig(PointConfig(("laa_segments", "lateral"), -1),
                            uacs=(LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2), 1 - LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2))),
}


# ==================================================================================================
PV_SEGMENTS = (1 / 4, 5 / 8)
LAA_HEIGHT = 9 / 10

parameterization_configs = {
    # ---------- PV boundaries ----------
    "PV": {
        "LIPV": {
            "inner": ParameterizationConfig(path=("PV", "LIPV", "inner"),
                                            markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                     ("PV", "LIPV", "inner", "septal_lateral"),
                                                     ("PV", "LIPV", "inner", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            "outer": ParameterizationConfig(path=("PV", "LIPV", "outer"),
                                            markers=(("PV", "LIPV", "outer", "anterior_posterior"),
                                                     ("PV", "LIPV", "outer", "septal_lateral"),
                                                     ("PV", "LIPV", "outer", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            },
        "LSPV": {
            "inner": ParameterizationConfig(path=("PV", "LSPV", "inner"),
                                            markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                     ("PV", "LSPV", "inner", "septal_lateral"),
                                                     ("PV", "LSPV", "inner", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            "outer": ParameterizationConfig(path=("PV", "LSPV", "outer"),
                                            markers=(("PV", "LSPV", "outer", "anterior_posterior"),
                                                     ("PV", "LSPV", "outer", "septal_lateral"),
                                                     ("PV", "LSPV", "outer", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            },
        "RSPV": {
            "inner": ParameterizationConfig(path=("PV", "RSPV", "inner"),
                                            markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                     ("PV", "RSPV", "inner", "septal_lateral"),
                                                     ("PV", "RSPV", "inner", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            "outer": ParameterizationConfig(path=("PV", "RSPV", "outer"),
                                            markers=(("PV", "RSPV", "outer", "anterior_posterior"),
                                                     ("PV", "RSPV", "outer", "septal_lateral"),
                                                     ("PV", "RSPV", "outer", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            },
        "RIPV": {
            "inner": ParameterizationConfig(path=("PV", "RIPV", "inner"),
                                            markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                     ("PV", "RIPV", "inner", "septal_lateral"),
                                                     ("PV", "RIPV", "inner", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            "outer": ParameterizationConfig(path=("PV", "RIPV", "outer"),
                                            markers=(("PV", "RIPV", "outer", "anterior_posterior"),
                                                     ("PV", "RIPV", "outer", "septal_lateral"),
                                                     ("PV", "RIPV", "outer", "diagonal")),
                                            marker_values=(0, PV_SEGMENTS[0], PV_SEGMENTS[1])),
            },
    },

    # ---------- PV segments ----------
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ParameterizationConfig(path=("pv_segments", "LIPV", "anterior_posterior"),
                                                         markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                                  ("PV", "LIPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(path=("pv_segments", "LIPV", "septal_lateral"),
                                                     markers=(("PV", "LIPV", "inner", "septal_lateral"),
                                                              ("PV", "LIPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(path=("pv_segments", "LIPV", "diagonal"),
                                               markers=(("PV", "LIPV", "inner", "diagonal"),
                                                        ("PV", "LIPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "LSPV": {
            "anterior_posterior": ParameterizationConfig(path=("pv_segments", "LSPV", "anterior_posterior"),
                                                         markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                                  ("PV", "LSPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(path=("pv_segments", "LSPV", "septal_lateral"),
                                                     markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                              ("PV", "LSPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(path=("pv_segments", "LSPV", "diagonal"),
                                               markers=(("PV", "LSPV", "inner", "diagonal"),
                                                        ("PV", "LSPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "RSPV": {
            "anterior_posterior": ParameterizationConfig(path=("pv_segments", "RSPV", "anterior_posterior"),
                                                         markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                                  ("PV", "RSPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(path=("pv_segments", "RSPV", "septal_lateral"),
                                                     markers=(("PV", "RSPV", "inner", "septal_lateral"),
                                                              ("PV", "RSPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(path=("pv_segments", "RSPV", "diagonal"),
                                               markers=(("PV", "RSPV", "inner", "diagonal"),
                                                        ("PV", "RSPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
        "RIPV": {
            "anterior_posterior": ParameterizationConfig(path=("pv_segments", "RIPV", "anterior_posterior"),
                                                         markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                                  ("PV", "RIPV", "outer", "anterior_posterior")),
                                                         marker_values=(0, 1)),
            "septal_lateral": ParameterizationConfig(path=("pv_segments", "RIPV", "septal_lateral"),
                                                     markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                              ("PV", "RIPV", "outer", "septal_lateral")),
                                                     marker_values=(0, 1)),
            "diagonal": ParameterizationConfig(path=("pv_segments", "RIPV", "diagonal"),
                                               markers=(("PV", "RIPV", "inner", "diagonal"),
                                                        ("PV", "RIPV", "outer", "diagonal")),
                                               marker_values=(0, 1)),
        },
    },

    # ---------- Roof ----------
    "roof": {
        "LIPV_LSPV": ParameterizationConfig(path=("roof", "LIPV_LSPV"),
                                            markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                     ("PV", "LSPV", "inner", "anterior_posterior")),
                                            marker_values=(0, 1)),
        "LSPV_RSPV": ParameterizationConfig(path=("roof", "LSPV_RSPV"),
                                            markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                     ("PV", "RSPV", "inner", "septal_lateral")),
                                            marker_values=(0, 1)),
        "RSPV_RIPV": ParameterizationConfig(path=("roof", "RSPV_RIPV"),
                                            markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                     ("PV", "RIPV", "inner", "anterior_posterior")),
                                            marker_values=(0, 1)),
        "RIPV_LIPV": ParameterizationConfig(path=("roof", "RIPV_LIPV"),
                                            markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                     ("PV", "LIPV", "inner", "septal_lateral")),
                                            marker_values=(0, 1)),
    },

    # ---------- Diagonals ----------
    "diagonal": {
        "LIPV_LAA": ParameterizationConfig(path=("diagonal", "LIPV_LAA"),
                                           markers=(("PV", "LIPV", "inner", "diagonal"), ("LAA", "LIPV")),
                                           marker_values=(0, 1)),
        "LAA_MV": ParameterizationConfig(path=("diagonal", "LAA_MV"),
                                         markers=(("LAA", "MV"), ("MV", "LAA")),
                                         marker_values=(0, 1)),
        "LSPV_MV": ParameterizationConfig(path=("diagonal", "LSPV_MV"),
                                          markers=(("PV", "LSPV", "inner", "diagonal"), ("LSPV_MV",), ("MV", "LSPV")),
                                          marker_values=(0, LAA_HEIGHT, 1)),
        "RSPV_MV": ParameterizationConfig(path=("diagonal", "RSPV_MV"),
                                          markers=(("PV", "RSPV", "inner", "diagonal"), ("MV", "RSPV")),
                                          marker_values=(0, 1)),
        "RIPV_MV": ParameterizationConfig(path=("diagonal", "RIPV_MV"),
                                          markers=(("PV", "RIPV", "inner", "diagonal"), ("RIPV_MV",), ("MV", "RIPV")),
                                          marker_values=(0, LAA_HEIGHT, 1)),
   },

    # ---------- LAA Segments ----------
    "laa_segments": {
        "posterior": ParameterizationConfig(path=("laa_segments", "posterior"),
                                            markers=(("LAA", "posterior"), ("RIPV_MV",)),
                                            marker_values=(0, 1)),
        "lateral": ParameterizationConfig(path=("laa_segments", "lateral"),
                                          markers=(("LAA", "lateral"), ("LSPV_MV",)),
                                          marker_values=(0, 1)),
    },

    # ---------- LAA and PV ----------
    "LAA": ParameterizationConfig(path=("LAA",),
                                  markers=(("LAA", "LIPV"), ("LAA", "lateral"), ("LAA", "MV"), ("LAA", "posterior")),
                                  marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
    "MV": ParameterizationConfig(path=("MV",),
                                 markers=(("MV", "RSPV"), ("MV", "LSPV"), ("MV", "LAA"), ("MV", "RIPV")),
                                 marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
}


# ==================================================================================================
patch_boundary_configs = {
    "roof": PatchBoundaryConfig(paths=(("roof", "LIPV_LSPV"), ("roof", "LSPV_RSPV"),
                                       ("roof", "RSPV_RIPV"), ("roof", "RIPV_LIPV"),
                                       ("PV", "LIPV", "inner"), ("PV", "LSPV", "inner"),
                                       ("PV", "RSPV", "inner"), ("PV", "RIPV", "inner")),
                                portions=((0, 1), (0, 1),
                                          (0, 1), (0, 1),
                                          (0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]),
                                          (0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]))),

    "anterior": PatchBoundaryConfig(paths=(("roof", "LSPV_RSPV"),
                                           ("diagonal", "LSPV_MV"), ("diagonal", "RSPV_MV"),
                                           ("PV", "LSPV", "inner"), ("PV", "RSPV", "inner"),
                                           ("MV",)),
                                    portions=((0, 1),
                                              (0, 1), (0, 1),
                                              (PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                              (0, PV_SEGMENTS[0]))),
    "septal": PatchBoundaryConfig(paths=(("roof", "RSPV_RIPV"),
                                         ("diagonal", "RIPV_MV"), ("diagonal", "RSPV_MV"),
                                         ("PV", "RIPV", "inner"), ("PV", "RSPV", "inner"),
                                         ("MV",)),
                                  portions=((0, 1),
                                            (0, 1), (0, 1),
                                            (PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                            (3 / 4, 1))),
    "posterior_roof": PatchBoundaryConfig(paths=(("roof", "RIPV_LIPV"),
                                                 ("diagonal", "RIPV_MV"), ("diagonal", "LIPV_LAA"),
                                                 ("PV", "RIPV", "inner"), ("PV", "LIPV", "inner"),
                                                 ("laa_segments", "posterior"), ("LAA",)),
                                    portions=((0, 1),
                                              (0, LAA_HEIGHT), (0, 1),
                                              (PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                              (0, 1), (3 / 4, 1))),
    "posterior_mv": PatchBoundaryConfig(paths=(("diagonal", "RIPV_MV"), ("diagonal", "LAA_MV"),
                                               ("laa_segments", "posterior"), ("LAA",),
                                               ("MV",)),
                                        portions=((LAA_HEIGHT, 1), (0, 1),
                                                  (0, 1), (1 / 2, 3 / 4),
                                                  (1 / 2, 3 / 4))),
    "lateral_roof": PatchBoundaryConfig(paths=(("roof", "LIPV_LSPV"),
                                               ("diagonal", "LIPV_LAA"), ("diagonal", "LSPV_MV"),
                                               ("PV", "LIPV", "inner"), ("PV", "LSPV", "inner"),
                                               ("laa_segments", "lateral"), ("LAA",)),
                                        portions=((0, 1),
                                                  (0, 1), (0, LAA_HEIGHT),
                                                  (PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                                  (0, 1),
                                                  (0, 1 / 4))),
    "lateral_mv": PatchBoundaryConfig(paths=(("diagonal", "LAA_MV"), ("diagonal", "LSPV_MV"),
                                             ("laa_segments", "lateral"), ("LAA",),
                                             ("MV",)),
                                      portions=((0, 1), (LAA_HEIGHT, 1),
                                                (0, 1), (1 / 4, 1 / 2),
                                                (1 / 4, 1 / 2))),
    "laa": PatchBoundaryConfig(paths=(("LAA",),), portions=((0, 1),)),
    "pv_segments": {
        "LIPV": {
            "segment_1": PatchBoundaryConfig(paths=(("PV", "LIPV", "inner"), ("PV", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "anterior_posterior"), ("pv_segments", "LIPV", "septal_lateral")),
                                             portions=((0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("PV", "LIPV", "inner"), ("PV", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "septal_lateral"), ("pv_segments", "LIPV", "diagonal")),
                                             portions=((PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("PV", "LIPV", "inner"), ("PV", "LIPV", "outer"),
                                                    ("pv_segments", "LIPV", "diagonal"), ("pv_segments", "LIPV", "anterior_posterior")),
                                             portions=((PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                                       (0, 1), (0, 1))),
        },
        "LSPV": {
            "segment_1": PatchBoundaryConfig(paths=(("PV", "LSPV", "inner"), ("PV", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "anterior_posterior"), ("pv_segments", "LSPV", "septal_lateral")),
                                             portions=((0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("PV", "LSPV", "inner"), ("PV", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "septal_lateral"), ("pv_segments", "LSPV", "diagonal")),
                                             portions=((PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("PV", "LSPV", "inner"), ("PV", "LSPV", "outer"),
                                                    ("pv_segments", "LSPV", "diagonal"), ("pv_segments", "LSPV", "anterior_posterior")),
                                             portions=((PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                                       (0, 1), (0, 1))),
        },
        "RSPV": {
            "segment_1": PatchBoundaryConfig(paths=(("PV", "RSPV", "inner"), ("PV", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "anterior_posterior"), ("pv_segments", "RSPV", "septal_lateral")),
                                             portions=((0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("PV", "RSPV", "inner"), ("PV", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "septal_lateral"), ("pv_segments", "RSPV", "diagonal")),
                                             portions=((PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("PV", "RSPV", "inner"), ("PV", "RSPV", "outer"),
                                                    ("pv_segments", "RSPV", "diagonal"), ("pv_segments", "RSPV", "anterior_posterior")),
                                             portions=((PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                                       (0, 1), (0, 1))),
        },
        "RIPV": {
            "segment_1": PatchBoundaryConfig(paths=(("PV", "RIPV", "inner"), ("PV", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "anterior_posterior"), ("pv_segments", "RIPV", "septal_lateral")),
                                             portions=((0, PV_SEGMENTS[0]), (0, PV_SEGMENTS[0]),
                                                       (0, 1), (0, 1))),
            "segment_2": PatchBoundaryConfig(paths=(("PV", "RIPV", "inner"), ("PV", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "septal_lateral"), ("pv_segments", "RIPV", "diagonal")),
                                             portions=((PV_SEGMENTS[0], PV_SEGMENTS[1]), (PV_SEGMENTS[0], PV_SEGMENTS[1]),
                                                       (0, 1), (0, 1))),
            "segment_3": PatchBoundaryConfig(paths=(("PV", "RIPV", "inner"), ("PV", "RIPV", "outer"),
                                                    ("pv_segments", "RIPV", "diagonal"), ("pv_segments", "RIPV", "anterior_posterior")),
                                             portions=((PV_SEGMENTS[1], 1), (PV_SEGMENTS[1], 1),
                                                       (0, 1), (0, 1))),
        },
    },
}


# ==================================================================================================
patch_configs = {
    "roof": PatchConfig(boundary=("roof",), outside=("MV",)),
    "anterior": PatchConfig(boundary=("anterior",), outside=("PV", "LIPV", "inner")),
    "septal": PatchConfig(boundary=("septal",), outside=("PV", "LIPV", "inner")),
    "posterior_roof": PatchConfig(boundary=("posterior_roof",), outside=("PV", "LSPV", "inner")),
    "posterior_mv": PatchConfig(boundary=("posterior_mv",), outside=("PV", "LSPV", "inner")),
    "lateral_roof": PatchConfig(boundary=("lateral_roof",), outside=("PV", "RIPV", "inner")),
    "lateral_mv": PatchConfig(boundary=("lateral_mv",), outside=("PV", "RIPV", "inner")),
    "laa": PatchConfig(boundary=("laa",), outside=("MV",)),
    "pv_segments": {
        "LIPV": {
            "segment_1": PatchConfig(boundary=("pv_segments", "LIPV", "segment_1"), outside=("MV",)),
            "segment_2": PatchConfig(boundary=("pv_segments", "LIPV", "segment_2"), outside=("MV",)),
            "segment_3": PatchConfig(boundary=("pv_segments", "LIPV", "segment_3"), outside=("MV",)),
        },
        "LSPV": {
            "segment_1": PatchConfig(boundary=("pv_segments", "LSPV", "segment_1"), outside=("MV",)),
            "segment_2": PatchConfig(boundary=("pv_segments", "LSPV", "segment_2"), outside=("MV",)),
            "segment_3": PatchConfig(boundary=("pv_segments", "LSPV", "segment_3"), outside=("MV",)),
        },
        "RSPV": {
            "segment_1": PatchConfig(boundary=("pv_segments", "RSPV", "segment_1"), outside=("MV",)),
            "segment_2": PatchConfig(boundary=("pv_segments", "RSPV", "segment_2"), outside=("MV",)),
            "segment_3": PatchConfig(boundary=("pv_segments", "RSPV", "segment_3"), outside=("MV",)),
        },
        "RIPV": {
            "segment_1": PatchConfig(boundary=("pv_segments", "RIPV", "segment_1"), outside=("MV",)),
            "segment_2": PatchConfig(boundary=("pv_segments", "RIPV", "segment_2"), outside=("MV",)),
            "segment_3": PatchConfig(boundary=("pv_segments", "RIPV", "segment_3"), outside=("MV",)),
        },
    }
}
