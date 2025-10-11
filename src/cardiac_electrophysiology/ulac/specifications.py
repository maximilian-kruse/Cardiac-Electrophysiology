import numpy as np

from . import base

# ==================================================================================================
# Paths
path_specs = {
    "roof": {
        "LIPV_LSPV": ("LIPV_inner", "LSPV_inner"),
        "LSPV_RSPV": ("LSPV_inner", "RSPV_inner"),
        "RSPV_RIPV": ("RSPV_inner", "RIPV_inner"),
        "RIPV_LIPV": ("RIPV_inner", "LIPV_inner"),
    },
    "diagonal": {
        "LIPV_LAA": ("LIPV_inner", "LAA"),
        "LAA_MV": ("LAA", "MV"),
        "LSPV_MV": ("LSPV_inner", "MV"),
        "RIPV_MV": ("RIPV_inner", "MV"),
        "RSPV_MV": ("RSPV_inner", "MV"),
    },
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": ("LIPV_LSPV", 0),
            "septal_lateral": ("RIPV_LIPV", -1),
            "diagonal": ("LIPV_LAA", 0),
        },
        "LSPV": {
            "anterior_posterior": ("LIPV_LSPV", -1),
            "septal_lateral": ("LSPV_RSPV", 0),
            "diagonal": ("LSPV_MV", 0),
        },
        "RSPV": {
            "anterior_posterior": ("RSPV_RIPV", 0),
            "septal_lateral": ("LSPV_RSPV", -1),
            "diagonal": ("RSPV_MV", 0),
        },
        "RIPV": {
            "anterior_posterior": ("RSPV_RIPV", -1),
            "septal_lateral": ("RIPV_LIPV", 0),
            "diagonal": ("RIPV_MV", 0),
        },
    },
}


# ==================================================================================================
# Parameterization
parameterization_specs = {
    "PV": {
        "start": "anterior_posterior",
        "markers": ["septal_lateral", "diagonal"],
        "marker_values": [1 / 4, 2 / 3],
    },
    "LAA": {
        "start": "LIPV",
        "markers": ["MV", "posterior"],
        "marker_values": [1 / 2, 3 / 4],
    },
    "MV": {
        "start": "RSPV",
        "markers": ["LSPV", "LAA", "RIPV"],
        "marker_values": [1 / 4, 1 / 2, 3 / 4],
    },
}


# ==================================================================================================
# UACs for elementary forms
PV_INNER_RADIUS = 0.1
PV_OUTER_RADIUS = 0.05
LAA_RADIUS = 0.1
LIPV_CENTER = (2 / 3, 2 / 3)
LSPV_CENTER = (2 / 3, 1 / 3)
RIPV_CENTER = (1 / 3, 2 / 3)
RSPV_CENTER = (1 / 3, 1 / 3)
LAA_CENTER = (5 / 6, 5 / 6)


uac_form_specs = {
    # ----- PV boundaries -----
    "pv_boundaries": {
        "LIPV_inner": base.UACCircle(
            center=LIPV_CENTER,
            radius=PV_INNER_RADIUS,
            orientation=-1,
            start_angle=3 / 2 * np.pi,
        ),
        "LSPV_inner": base.UACCircle(
            center=LSPV_CENTER,
            radius=PV_INNER_RADIUS,
            orientation=+1,
            start_angle=1 / 2 * np.pi,
        ),
        "RIPV_inner": base.UACCircle(
            center=RIPV_CENTER,
            radius=PV_INNER_RADIUS,
            orientation=+1,
            start_angle=3 / 2 * np.pi,
        ),
        "RSPV_inner": base.UACCircle(
            center=RSPV_CENTER,
            radius=PV_INNER_RADIUS,
            orientation=-1,
            start_angle=1 / 2 * np.pi,
        ),
        "LIPV_outer": base.UACCircle(
            center=LIPV_CENTER,
            radius=PV_OUTER_RADIUS,
            orientation=-1,
            start_angle=3 / 2 * np.pi,
        ),
        "LSPV_outer": base.UACCircle(
            center=LSPV_CENTER,
            radius=PV_OUTER_RADIUS,
            orientation=+1,
            start_angle=1 / 2 * np.pi,
        ),
        "RIPV_outer": base.UACCircle(
            center=RIPV_CENTER,
            radius=PV_OUTER_RADIUS,
            orientation=+1,
            start_angle=3 / 2 * np.pi,
        ),
        "RSPV_outer": base.UACCircle(
            center=RSPV_CENTER,
            radius=PV_OUTER_RADIUS,
            orientation=-1,
            start_angle=1 / 2 * np.pi,
        ),
    },
    # ----- LAA boundary -----
    "laa_boundary": base.UACCircle(
        center=LAA_CENTER,
        radius=LAA_RADIUS,
        orientation=+1,
        start_angle=5 / 4 * np.pi,
    ),
    # ----- MV boundary -----
    "mv_boundary": base.UACRectangle(
        lower_left_corner=(0, 0),
        length_alpha=1,
        length_beta=1,
    ),
    # ----- PV segments -----
    "pv_segments": {
        "LIPV": {
            "anterior_posterior": base.UACLine(
                start=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                end=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ),
            "septal_lateral": base.UACLine(
                start=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                end=(LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
            ),
            "diagonal": base.UACLine(
                start=(
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                end=(
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "LSPV": {
            "anterior_posterior": base.UACLine(
                start=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                end=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ),
            "septal_lateral": base.UACLine(
                start=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                end=(LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
            ),
            "diagonal": base.UACLine(
                start=(
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                end=(
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "RSPV": {
            "anterior_posterior": base.UACLine(
                start=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                end=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ),
            "septal_lateral": base.UACLine(
                start=(RSPV_CENTER[0] - PV_INNER_RADIUS, RSPV_CENTER[1]),
                end=(RSPV_CENTER[0] - PV_OUTER_RADIUS, RSPV_CENTER[1]),
            ),
            "diagonal": base.UACLine(
                start=(
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                end=(
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
        "RIPV": {
            "anterior_posterior": base.UACLine(
                start=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                end=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ),
            "septal_lateral": base.UACLine(
                start=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                end=(RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
            ),
            "diagonal": base.UACLine(
                start=(
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                end=(
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ),
        },
    },
    # ----- Roof -----
    "roof": {
        "LIPV_LSPV": base.UACLine(
            start=(LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
            end=(LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
        ),
        "LSPV_RSPV": base.UACLine(
            start=(LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
            end=(RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
        ),
        "RSPV_RIPV": base.UACLine(
            start=(RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
            end=(RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
        ),
        "RIPV_LIPV": base.UACLine(
            start=(RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
            end=(LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
        ),
    },
    # ----- Diagonals -----
    "diagonal": {
        "LIPV_LAA": base.UACLine(
            start=(
                LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
            ),
            end=(
                LAA_CENTER[0] - LAA_RADIUS / np.sqrt(2),
                LAA_CENTER[1] - LAA_RADIUS / np.sqrt(2),
            ),
        ),
        "LAA_MV": base.UACLine(
            start=(
                LAA_CENTER[0] + LAA_RADIUS / np.sqrt(2),
                LAA_CENTER[1] + LAA_RADIUS / np.sqrt(2),
            ),
            end=(1, 1),
        ),
        "LSPV_MV": base.UACLine(
            start=(
                LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
            ),
            end=(1, 0),
        ),
        "RSPV_MV": base.UACLine(
            start=(
                RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
            ),
            end=(0, 0),
        ),
        "RIPV_MV": base.UACLine(
            start=(
                RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
            ),
            end=(0, 1),
        ),
    },
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
            "LIPV_LSPV": (0, 1),
        },
        "diagonal": {
            "LSPV_MV": (0, 1),
            "RSPV_MV": (0, 1),
        },
        "pv_boundaries": {
            "LSPV_inner": (1 / 4, 2 / 3),
            "RSPV_inner": (1 / 4, 2 / 3),
        },
        "mv_boundary": (0, 0.25),
    },
    # ----- Posterior -----
    "posterior": {
        "roof": {
            "RIPV_LIPV": (0, 1),
        },
        "diagonal": {
            "RIPV_MV": (0, 1),
            "LIPV_LAA": (0, 1),
            "LAA_MV": (0, 1),
        },
        "pv_boundaries": {
            "LIPV_inner": (1 / 4, 2 / 3),
            "RIPV_inner": (1 / 4, 2 / 3),
        },
        "mv_boundary": (0.5, 0.75),
        "laa_boundary": (0.5, 1),
    },
    # ----- Septal -----
    "septal": {
        "roof": {
            "RSPV_RIPV": (0, 1),
        },
        "diagonal": {
            "RIPV_MV": (0, 1),
            "RSPV_MV": (0, 1),
        },
        "pv_boundaries": {
            "RIPV_inner": (2 / 3, 1),
            "RSPV_inner": (2 / 3, 1),
        },
        "mv_boundary": (0.75, 1),
    },
    # ----- Lateral -----
    "lateral": {
        "roof": {
            "LIPV_LSPV": (0, 1),
        },
        "diagonal": {
            "LIPV_LAA": (0, 1),
            "LAA_MV": (0, 1),
            "LSPV_MV": (0, 1),
        },
        "pv_boundaries": {
            "LIPV_inner": (2 / 3, 1),
            "LSPV_inner": (2 / 3, 1),
        },
        "mv_boundary": (0.25, 0.5),
    },
    # ----- PV Segments -----
    "pv_segments": {
        "segment_1": {
            "boundary": (0, 1/4),
            "segments": {
                "anterior_posterior": (0, 1),
                "septal_lateral": (0, 1),
            }
        },
        "segment_2": {
            "boundary": (1/4, 2/3),
            "segments": {
                "septal_lateral": (0, 1),
                "diagonal": (0, 1),
            }
        },
        "segment_3": {
            "boundary": (2/3, 1),
            "segments": {
                "diagonal": (0, 1),
                "anterior_posterior": (0, 1),
            }
        },
    },
}
