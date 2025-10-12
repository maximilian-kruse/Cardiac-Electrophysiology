from dataclasses import dataclass

import numpy as np

# ruff: noqa: E501


# ==================================================================================================
@dataclass
class PathSpec:
    type: str
    name: tuple[str, ...]


@dataclass
class PointSpec:
    containing_path: str
    index: int


@dataclass
class BoundaryPathSpec:
    feature_tag: str
    coincides_with_mesh_boundary: bool


@dataclass
class ConnectionPathSpec:
    type: str
    start: str | PointSpec
    end: str | PointSpec


@dataclass
class MarkerSpec:
    point: PointSpec
    uacs: tuple[int, int]


@dataclass
class ParameterizationSpec:
    markers: tuple[str]
    marker_values: tuple[float]


# ==================================================================================================
boundary_path_specs = {
    "PV": {
        "LIPV": {
            "inner": BoundaryPathSpec("LIPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathSpec("LIPV", coincides_with_mesh_boundary=True)
            },
        "LSPV": {
            "inner": BoundaryPathSpec("LSPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathSpec("LSPV", coincides_with_mesh_boundary=True)
        },
        "RSPV": {
            "inner": BoundaryPathSpec("RSPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathSpec("RSPV", coincides_with_mesh_boundary=True)
        },
        "RIPV": {
            "inner": BoundaryPathSpec("RIPV", coincides_with_mesh_boundary=False),
            "outer": BoundaryPathSpec("RIPV", coincides_with_mesh_boundary=True)
        },
    },
    "LAA": BoundaryPathSpec("LAA", coincides_with_mesh_boundary=False),
    "MV": BoundaryPathSpec("MV", coincides_with_mesh_boundary=True),
}


# ==================================================================================================
direct_connection_path_specs = {
    "roof": {
        "LIPV_LSPV": ConnectionPathSpec(start=("PV", "LIPV", "inner"), end=("PV", "LSPV", "inner")),
        "LSPV_RSPV": ConnectionPathSpec(start=("PV", "LSPV", "inner"), end=("PV", "RSPV", "inner")),
        "RSPV_RIPV": ConnectionPathSpec(start=("PV", "RSPV", "inner"), end=("PV", "RIPV", "inner")),
        "RIPV_LIPV": ConnectionPathSpec(start=("PV", "RIPV", "inner"), end=("PV", "LIPV", "inner")),
    },
    "diagonal": {
        "LAA_MV": ConnectionPathSpec(start=("LAA",), end=("MV",)),
        "LIPV_LAA": ConnectionPathSpec(start=("PV", "LIPV", "inner"), end=("LAA",)),
        "LSPV_MV": ConnectionPathSpec(start=("PV", "LSPV", "inner"), end=("MV",)),
        "RIPV_MV": ConnectionPathSpec(start=("PV", "RIPV", "inner"), end=("MV",)),
        "RSPV_MV": ConnectionPathSpec(start=("PV", "RSPV", "inner"), end=("MV",)),
    },
}


# ==================================================================================================
indirect_connection_path_specs = {
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ConnectionPathSpec(start=PointSpec(("roof", "LIPV_LSPV"), 0), end=("PV", "LIPV", "outer")),
            "septal_lateral": ConnectionPathSpec(start=PointSpec(("roof", "RIPV_LIPV"), -1), end=("PV", "LIPV", "outer")),
            "diagonal": ConnectionPathSpec(start=PointSpec(("diagonal", "LIPV_LAA"), 0), end=("PV", "LIPV", "outer")),
        },
        "LSPV": {
            "anterior_posterior": ConnectionPathSpec(start=PointSpec(("roof", "LIPV_LSPV"), -1), end=("PV", "LSPV", "inner")),
            "septal_lateral": ConnectionPathSpec(start=PointSpec(("roof", "RIPV_LIPV"), -1), end=("PV", "LSPV", "inner")),
            "diagonal": ConnectionPathSpec(start=PointSpec(("diagonal", "LSPV_MV"), 0), end=("PV", "LSPV", "inner")),
        },
        "RSPV": {
            "anterior_posterior": ConnectionPathSpec(start=PointSpec(("roof", "RSPV_RIPV"), 0), end=("PV", "RSPV", "inner")),
            "septal_lateral": ConnectionPathSpec(start=PointSpec(("roof", "LSPV_RSPV"), -1), end=("PV", "RSPV", "inner")),
            "diagonal": ConnectionPathSpec(start=PointSpec(("diagonal", "RSPV_MV"), 0), end=("PV", "RSPV", "inner")),
        },
        "RIPV": {
            "anterior_posterior": ConnectionPathSpec(start=PointSpec(("roof", "RSPV_RIPV"), index=-1), end=("PV", "RIPV", "inner")),
            "septal_lateral": ConnectionPathSpec(start=PointSpec(("roof", "RIPV_LIPV"), index=0), end=("PV", "RIPV", "inner")),
            "diagonal": ConnectionPathSpec(start=PointSpec(("diagonal", "RIPV_MV"), index=0), end=("PV", "RIPV", "inner")),
        },
    },
    "laa_segments": {
        "posterior": ConnectionPathSpec(start="LAA", end=("diagonal", "RIPV_MV")),
        "lateral": ConnectionPathSpec(start=("LAA"), end=("diagonal", "LSPV_MV")),
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

marker_specs = {
    # ---------- PVs ----------
    "PV": {
        "LIPV": {
            "inner": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "lipv", "anterior_posterior"), 0),
                                                 uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "lipv", "septal_lateral"), 0),
                                             uacs=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "lipv", "diagonal"), 0),
                                       uacs=(LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "lipv", "anterior_posterior"), -1),
                                                 uacs=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "lipv", "septal_lateral"), -1),
                                             uacs=(LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "lipv", "diagonal"), -1),
                                       uacs=(LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "LSPV": {
            "inner": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "lspv", "anterior_posterior"), 0),
                                                 uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "lspv", "septal_lateral"), 0),
                                             uacs=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "lspv", "diagonal"), 0),
                                       uacs=(LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "lspv", "anterior_posterior"), -1),
                                                 uacs=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "lspv", "septal_lateral"), -1),
                                             uacs=(LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "lspv", "diagonal"), -1),
                                       uacs=(LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2), LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RSPV": {
            "inner": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "rspv", "anterior_posterior"), 0),
                                                 uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "rspv", "septal_lateral"), 0),
                                             uacs=(RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "rspv", "diagonal"), 0),
                                       uacs=(RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "rspv", "anterior_posterior"), -1),
                                                 uacs=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "rspv", "septal_lateral"), -1),
                                             uacs=(RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "rspv", "diagonal"), -1),
                                       uacs=(RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
        "RIPV": {
            "inner": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "ripv", "anterior_posterior"), 0),
                                                 uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "ripv", "septal_lateral"), 0),
                                             uacs=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "ripv", "diagonal"), 0),
                                       uacs=(RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2))),
            },
            "outer": {
                "anterior_posterior": MarkerSpec(PointSpec(("pv_segments", "ripv", "anterior_posterior"), -1),
                                                 uacs=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS)),
                "septal_lateral": MarkerSpec(PointSpec(("pv_segments", "ripv", "septal_lateral"), -1),
                                             uacs=(RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1])),
                "diagonal": MarkerSpec(PointSpec(("pv_segments", "ripv", "diagonal"), -1),
                                       uacs=(RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2))),
            },
        },
    },

    # ---------- LAA ----------
    "LAA": {
        "LIPV": MarkerSpec(PointSpec(("diagonal", "LIPV_LAA"), -1),
                           uacs=((LAA_CENTER[0] - LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] - LAA_RADIUS) / np.sqrt(2))),
        "MV": MarkerSpec(PointSpec(("diagonal", "LIPV_LAA"), 0),
                         uacs=((LAA_CENTER[0] + LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] + LAA_RADIUS) / np.sqrt(2))),
        "posterior": MarkerSpec(PointSpec(("laa_segments", "posterior"), 0),
                                uacs=((LAA_CENTER[0] - LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] + LAA_RADIUS) / np.sqrt(2))),
        "lateral": MarkerSpec(PointSpec(("laa_segments", "lateral"), 0),
                             uacs=((LAA_CENTER[0] + LAA_RADIUS) / np.sqrt(2), (LAA_CENTER[1] - LAA_RADIUS) / np.sqrt(2))),
    },

    # ---------- MV ----------
    "MV": {
        "LAA": MarkerSpec(PointSpec(("diagonal","LAA_MV"), -1), uacs=(1, 1)),
        "LSPV": MarkerSpec(PointSpec(("diagonal", "LSPV_MV"), -1), uacs=(1, 0)),
        "RSPV": MarkerSpec(PointSpec(("diagonal", "RSPV_MV"), -1), uacs=(0, 0)),
        "RIPV": MarkerSpec(PointSpec(("diagonal", "RIPV_MV"), -1), uacs=(0, 1)),
    },
    "RIPV_MV": MarkerSpec(PointSpec(("laa_segments", "posterior"), -1),
                          uacs=(RIPV_CENTER[0] / 2, (1 + RIPV_CENTER[1]) / 2)),
    "LSPV_MV": MarkerSpec(PointSpec(("laa_segments", "posterior"), -1),
                          uacs=((1 + LSPV_CENTER[0]) / 2, LSPV_CENTER[1] / 2)),
}


# ==================================================================================================
# Parameterization
parameterization_specs = {
    # ---------- PV boundaries ----------
    "pv_boundaries": {
        "LIPV": {
            "inner": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                   ("PV", "LIPV", "inner", "septal_lateral"),
                                                   ("PV", "LIPV", "inner", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationSpec(markers=(("PV", "LIPV", "outer", "anterior_posterior"),
                                                   ("PV", "LIPV", "outer", "septal_lateral"),
                                                   ("PV", "LIPV", "outer", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            },
        "LSPV": {
            "inner": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                   ("PV", "LSPV", "inner", "septal_lateral"),
                                                   ("PV", "LSPV", "inner", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationSpec(markers=(("PV", "LSPV", "outer", "anterior_posterior"),
                                                   ("PV", "LSPV", "outer", "septal_lateral"),
                                                   ("PV", "LSPV", "outer", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            },
        "RSPV": {
            "inner": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                   ("PV", "RSPV", "inner", "septal_lateral"),
                                                   ("PV", "RSPV", "inner", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationSpec(markers=(("PV", "RSPV", "outer", "anterior_posterior"),
                                                   ("PV", "RSPV", "outer", "septal_lateral"),
                                                   ("PV", "RSPV", "outer", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            },
        "RIPV": {
            "inner": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                   ("PV", "RIPV", "inner", "septal_lateral"),
                                                   ("PV", "RIPV", "inner", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            "outer": ParameterizationSpec(markers=(("PV", "RIPV", "outer", "anterior_posterior"),
                                                   ("PV", "RIPV", "outer", "septal_lateral"),
                                                   ("PV", "RIPV", "outer", "diagonal")),
                                          marker_values=(0, 1 / 3, 2 / 3)),
            },
    },

    # ---------- PV segments ----------
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                                ("PV", "LIPV", "outer", "anterior_posterior")),
                                                       marker_values=(0, 1)),
            "septal_lateral": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "septal_lateral"),
                                                            ("PV", "LIPV", "outer", "septal_lateral")),
                                                   marker_values=(0, 1)),
            "diagonal": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "diagonal"),
                                                      ("PV", "LIPV", "outer", "diagonal")),
                                             marker_values=(0, 1)),
        },
        "LSPV": {
            "anterior_posterior": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "anterior_posterior"),
                                                                ("PV", "LSPV", "outer", "anterior_posterior")),
                                                       marker_values=(0, 1)),
            "septal_lateral": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                            ("PV", "LSPV", "outer", "septal_lateral")),
                                                   marker_values=(0, 1)),
            "diagonal": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "diagonal"),
                                                      ("PV", "LSPV", "outer", "diagonal")),
                                             marker_values=(0, 1)),
        },
        "RSPV": {
            "anterior_posterior": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                                ("PV", "RSPV", "outer", "anterior_posterior")),
                                                       marker_values=(0, 1)),
            "septal_lateral": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "septal_lateral"),
                                                            ("PV", "RSPV", "outer", "septal_lateral")),
                                                   marker_values=(0, 1)),
            "diagonal": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "diagonal"),
                                                      ("PV", "RSPV", "outer", "diagonal")),
                                             marker_values=(0, 1)),
        },
        "RIPV": {
            "anterior_posterior": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "anterior_posterior"),
                                                                ("PV", "RIPV", "outer", "anterior_posterior")),
                                                       marker_values=(0, 1)),
            "septal_lateral": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                            ("PV", "RIPV", "outer", "septal_lateral")),
                                                   marker_values=(0, 1)),
            "diagonal": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "diagonal"),
                                                      ("PV", "RIPV", "outer", "diagonal")),
                                             marker_values=(0, 1)),
        },
    },

    # ---------- Roof ----------
    "roof": {
        "LIPV_LSPV": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "anterior_posterior"),
                                                   ("PV", "LSPV", "inner", "anterior_posterior")),
                                          marker_values=(0, 1)),
        "LSPV_RSPV": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "septal_lateral"),
                                                   ("PV", "RSPV", "inner", "septal_lateral")),
                                          marker_values=(0, 1)),
        "RSPV_RIPV": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "anterior_posterior"),
                                                   ("PV", "RIPV", "inner", "anterior_posterior")),
                                          marker_values=(0, 1)),
        "RIPV_LIPV": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "septal_lateral"),
                                                   ("PV", "LIPV", "inner", "septal_lateral")),
                                          marker_values=(0, 1)),
    },

    # ---------- Diagonals ----------
    "diagonal": {
        "LIPV_LAA": ParameterizationSpec(markers=(("PV", "LIPV", "inner", "diagonal"), ("LAA", "LIPV")),
                                         marker_values=(0, 1)),
        "LAA_MV": ParameterizationSpec(markers=(("LAA", "LIPV"), ("MV", "LAA")),
                                       marker_values=(0, 1)),
        "LSPV_MV": ParameterizationSpec(markers=(("PV", "LSPV", "inner", "diagonal"), ("LSPV_MV",), ("MV", "LSPV")),
                                        marker_values=(0, 1 / 2, 1)),
        "RSPV_MV": ParameterizationSpec(markers=(("PV", "RSPV", "inner", "diagonal"), ("MV", "RSPV")),
                                        marker_values=(0, 1)),
        "RIPV_MV": ParameterizationSpec(markers=(("PV", "RIPV", "inner", "diagonal"), ("LSPV_MV",), ("MV", "RIPV")),
                                        marker_values=(0, 1 / 2, 1)),
    },

    # ---------- LAA Segments ----------
    "laa_segments": {
        "posterior": ParameterizationSpec(markers=(("LAA", "posterior"), ("RIPV_MV",)),
                                          marker_values=(0, 1)),
        "lateral": ParameterizationSpec(markers=(("LAA", "lateral"), ("LSPV_MV",)),
                                        marker_values=(0, 1)),
    },

    # ---------- LAA and PV ----------
    "LAA": ParameterizationSpec(markers=(("LAA", "LIPV"), ("LAA", "lateral"), ("LAA", "MV"), ("LAA", "posterior")),
                                marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
    "MV": ParameterizationSpec(markers=(("MV", "RSPV"), ("MV", "LSPV"), ("MV", "LAA"), ("MV", "RIPV")),
                               marker_values=(0, 1 / 4, 1 / 2, 3 / 4)),
}


# ==================================================================================================
# Mesh Patch boundaries
patch_boundary_specs = {
    # ----- Roof -----
    "roof": {
        "roof": {
            "LIPV_LSPV": (0, 1),
            "LSPV_RSPV": (0, 1),
            "RSPV_RIPV": (0, 1),
            "RIPV_LIPV": (0, 1),
        },
        "pv_boundaries": {
            "LIPV_inner": (0, 1 / 4),
            "LSPV_inner": (0, 1 / 4),
            "RSPV_inner": (0, 1 / 4),
            "RIPV_inner": (0, 1 / 4),
        },
    },

    # ----- Anterior -----
    "anterior": {
        "roof": {
            "LSPV_RSPV": (0, 1),
        },
        "diagonal": {
            "LSPV_MV": (0, 1),
            "RSPV_MV": (0, 1),
        },
        "pv_boundaries": {
            "LSPV_inner": (1 / 4, 5 / 8),
            "RSPV_inner": (1 / 4, 5 / 8),
        },
        "mv_boundary": (0, 1 / 4),
    },

    # ----- Septal-Roof -----
    "septal": {
        "roof": {
            "RSPV_RIPV": (0, 1),
        },
        "diagonal": {
            "RIPV_MV": (0, 1),
            "RSPV_MV": (0, 1),
        },
        "pv_boundaries": {
            "RIPV_inner": (5 / 8, 1),
            "RSPV_inner": (5 / 8, 1),
        },
        "mv_boundary": (3 / 4, 1),
    },

    # ----- Posterior-Roof -----
    "posterior_roof": {
        "roof": {
            "RIPV_LIPV": (0, 1),
        },
        "diagonal": {
            "RIPV_MV": (0, 1 / 2),
            "LIPV_LAA": (0, 1),
        },
        "pv_boundaries": {
            "LIPV_inner": (1 / 4, 5 / 8),
            "RIPV_inner": (1 / 4, 5 / 8),
        },
        "laa_segments": {
            "posterior": (0, 1)
        },
        "laa_boundary": (3 / 4, 1),
    },

    # ----- Posterior-MV -----
    "posterior_mv": {
        "diagonal": {
            "RIPV_MV": (1 / 2, 1),
            "LAA_MV": (0, 1),
        },
        "laa_segments": {
            "posterior": (0, 1)
        },
        "laa_boundary": (1 / 2, 3 / 4),
        "mv_boundary": (1 / 2, 3 / 4)
    },

    # ----- Lateral-Roof -----
    "lateral_roof": {
        "roof": {
            "LIPV_LSPV": (0, 1),
        },
        "diagonal": {
            "LIPV_LAA": (0, 1),
            "LSPV_MV": (0, 1 / 2),
        },
        "pv_boundaries": {
            "LIPV_inner": (5 / 8, 1),
            "LSPV_inner": (5 / 8, 1),
        },
        "laa_segments": {
            "lateral": (0, 1)
        },
        "laa_boundary": (0, 1 / 4),
    },

    # ----- Lateral-MV -----
    "lateral_mv": {
        "diagonal": {
            "LAA_MV": (0, 1),
            "LSPV_MV": (1 / 2, 1),
        },
        "laa_segments": {
            "lateral": (0, 1)
        },
        "laa_boundary": (1 / 4, 1 / 2),
    },

    # ----- PV Segments -----
    "pv_segments": {
        "segment_1": {
            "boundary": (0, 1 / 4),
            "segments": {
                "anterior_posterior": (0, 1),
                "septal_lateral": (0, 1),
            },
        },
        "segment_2": {
            "boundary": (1 / 4, 5 / 8),
            "segments": {
                "septal_lateral": (0, 1),
                "diagonal": (0, 1),
            },
        },
        "segment_3": {
            "boundary": (5 / 8, 1),
            "segments": {
                "diagonal": (0, 1),
                "anterior_posterior": (0, 1),
            },
        },
    },
}
