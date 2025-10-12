from dataclasses import dataclass

import numpy as np

from . import base


# ==================================================================================================
# Feature Tags
@dataclass
class FeatureTags:
    MV: int = None
    LAA: int = None
    LIPV: int = None
    LSPV: int = None
    RIPV: int = None
    RSPV: int = None


# ==================================================================================================
# Paths
@dataclass
class PVBoundaryPaths:
    LIPV_inner: np.ndarray = None
    LSPV_inner: np.ndarray = None
    RIPV_inner: np.ndarray = None
    RSPV_inner: np.ndarray = None
    LIPV_outer: np.ndarray = None
    LSPV_outer: np.ndarray = None
    RIPV_outer: np.ndarray = None
    RSPV_outer: np.ndarray = None


@dataclass
class RoofPaths:
    LIPV_LSPV: np.ndarray = None
    LSPV_RSPV: np.ndarray = None
    RSPV_RIPV: np.ndarray = None
    RIPV_LIPV: np.ndarray = None


@dataclass
class DiagonalPaths:
    LIPV_LAA: np.ndarray = None
    LAA_MV: np.ndarray = None
    LSPV_MV: np.ndarray = None
    RIPV_MV: np.ndarray = None
    RSPV_MV: np.ndarray = None


@dataclass
class Segments:
    anterior_posterior: float = None
    septal_lateral: float = None
    diagonal: float = None


@dataclass
class PVSegmentPaths:
    LIPV: Segments = None
    LSPV: Segments = None
    RIPV: Segments = None
    RSPV: Segments = None


@dataclass
class Paths:
    pv_boundaries: PVBoundaryPaths = None
    laa_boundary: np.ndarray = None
    mv_boundary: np.ndarray = None
    pv_segments: PVSegmentPaths = None
    roof: RoofPaths = None
    diagonal: DiagonalPaths = None
    laa_posterior: np.ndarray = None


# ==================================================================================================
# Markers
@dataclass
class SegmentMarkers:
    anterior_posterior: float = None
    septal_lateral: float = None
    diagonal: float = None


@dataclass
class PVSegmentMarkers:
    LIPV_inner: SegmentMarkers = None
    LIPV_outer: SegmentMarkers = None
    LSPV_inner: SegmentMarkers = None
    LSPV_outer: SegmentMarkers = None
    RIPV_inner: SegmentMarkers = None
    RIPV_outer: SegmentMarkers = None
    RSPV_inner: SegmentMarkers = None
    RSPV_outer: SegmentMarkers = None


@dataclass
class LAAMarkers:
    LIPV: float = None
    MV: float = None
    posterior: float = None


@dataclass
class MVMarkers:
    LAA: float = None
    LSPV: float = None
    RIPV: float = None
    RSPV: float = None


@dataclass
class Markers:
    pv_segments: PVSegmentMarkers = None
    laa: LAAMarkers = None
    mv: MVMarkers = None


# ==================================================================================================
# Parameterized Paths
@dataclass
class ParameterizedSegments:
    anterior_posterior: base.ParameterizedPath = None
    septal_lateral: base.ParameterizedPath = None
    diagonal: base.ParameterizedPath = None


@dataclass
class ParameterizedPVSegmentPaths:
    LIPV: ParameterizedSegments = None
    LSPV: ParameterizedSegments = None
    RIPV: ParameterizedSegments = None
    RSPV: ParameterizedSegments = None


@dataclass
class ParameterizedPVBoundaryPaths:
    LIPV_inner: base.ParameterizedPath = None
    LSPV_inner: base.ParameterizedPath = None
    RIPV_inner: base.ParameterizedPath = None
    RSPV_inner: base.ParameterizedPath = None
    LIPV_outer: base.ParameterizedPath = None
    LSPV_outer: base.ParameterizedPath = None
    RIPV_outer: base.ParameterizedPath = None
    RSPV_outer: base.ParameterizedPath = None


@dataclass
class ParameterizedRoofPaths:
    LIPV_LSPV: base.ParameterizedPath = None
    LSPV_RSPV: base.ParameterizedPath = None
    RSPV_RIPV: base.ParameterizedPath = None
    RIPV_LIPV: base.ParameterizedPath = None


@dataclass
class ParameterizedDiagonalPaths:
    LIPV_LAA: base.ParameterizedPath = None
    LAA_MV: base.ParameterizedPath = None
    LSPV_MV: base.ParameterizedPath = None
    RIPV_MV: base.ParameterizedPath = None
    RSPV_MV: base.ParameterizedPath = None


@dataclass
class ParameterizedPaths:
    pv_boundaries: ParameterizedPVBoundaryPaths = None
    laa_boundary: base.ParameterizedPath = None
    mv_boundary: base.ParameterizedPath = None
    pv_segments: ParameterizedPVSegmentPaths = None
    roof: ParameterizedRoofPaths = None
    diagonal: ParameterizedDiagonalPaths = None


# ==================================================================================================
# UAC Paths
@dataclass
class UACSegments:
    anterior_posterior: base.UACPath = None
    septal_lateral: base.UACPath = None
    diagonal: base.UACPath = None


@dataclass
class UACPVSegmentPaths:
    LIPV: UACSegments = None
    LSPV: UACSegments = None
    RIPV: UACSegments = None
    RSPV: UACSegments = None


@dataclass
class UACPVBoundaryPaths:
    LIPV_inner: base.UACPath = None
    LSPV_inner: base.UACPath = None
    RIPV_inner: base.UACPath = None
    RSPV_inner: base.UACPath = None
    LIPV_outer: base.UACPath = None
    LSPV_outer: base.UACPath = None
    RIPV_outer: base.UACPath = None
    RSPV_outer: base.UACPath = None


@dataclass
class UACRoofPaths:
    LIPV_LSPV: base.UACPath = None
    LSPV_RSPV: base.UACPath = None
    RSPV_RIPV: base.UACPath = None
    RIPV_LIPV: base.UACPath = None


@dataclass
class UACDiagonalPaths:
    LIPV_LAA: base.UACPath = None
    LAA_MV: base.UACPath = None
    LSPV_MV: base.UACPath = None
    RIPV_MV: base.UACPath = None
    RSPV_MV: base.UACPath = None


@dataclass
class UACPaths:
    pv_boundaries: UACPVBoundaryPaths = None
    laa_boundary: base.UACPath = None
    mv_boundary: base.UACPath = None
    pv_segments: UACPVSegmentPaths = None
    roof: UACRoofPaths = None
    diagonal: UACDiagonalPaths = None


# ==================================================================================================
# Submesh Patch Boundaries
@dataclass
class PVSegmentPatchBoundaries:
    LIPV: UACSegments = None
    LSPV: UACSegments = None
    RIPV: UACSegments = None
    RSPV: UACSegments = None

@dataclass
class PatchBoundaries:
    pv_segments: PVSegmentPatchBoundaries = None
    laa: base.UACPath = None
    roof: base.UACPath = None
    anterior: base.UACPath = None
    posterior: base.UACPath = None
    septal: base.UACPath = None
    lateral: base.UACPath = None


# ==================================================================================================
# Submesh Patches
@dataclass
class SegmentPatches:
    segment_1: base.Submesh = None
    segment_2: base.Submesh = None
    segment_3: base.Submesh = None


@dataclass
class PVSegmentPatches:
    LIPV: SegmentPatches = None
    LSPV: SegmentPatches = None
    RIPV: SegmentPatches = None
    RSPV: SegmentPatches = None


@dataclass
class Patches:
    pv_segments: PVSegmentPatches = None
    laa: base.Submesh = None
    roof: base.Submesh = None
    anterior: base.Submesh = None
    posterior: base.Submesh = None
    septal: base.Submesh = None
    lateral: base.Submesh = None
