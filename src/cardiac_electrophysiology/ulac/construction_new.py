import operator
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pyvista as pv

from . import config_new as config
from . import construction_base as base
from . import workflow

# ==================================================================================================
type MarkerDict = dict[str, "MarkerDict" | base.Marker]
type PathDict = dict[str, "PathDict" | np.ndarray[tuple[int], np.dtype[np.float64]]]
type ParameterizedPathDict = dict[str, "ParameterizedPathDict" | base.ParameterizedPath]
type UACPathDict = dict[str, "UACPathDict" | base.UACPath]

type MarkerConfigDict = dict[str, "MarkerConfigDict" | config.MarkerConfig]
type PathConfigDict = dict[
    str, "PathConfigDict" | config.BoundaryPathConfig | config.ConnectionPathConfig
]
type AnyDict = (
    MarkerDict | PathDict | ParameterizedPathDict | UACPathDict | MarkerConfigDict | PathConfigDict
)


# --------------------------------------------------------------------------------------------------
def create_empty_dict_from_keys(input_dict: AnyDict) -> dict:
    def _create_empty_dict(d: AnyDict) -> dict:
        if isinstance(d, dict):
            return {key: _create_empty_dict(value) for key, value in d.items()}
        return None

    return _create_empty_dict(input_dict)


# --------------------------------------------------------------------------------------------------
def get_dict_entry(key_sequence: Iterable[str], data_dict: AnyDict) -> object:
    try:
        value = reduce(operator.getitem, key_sequence, data_dict)
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e
    return value


# --------------------------------------------------------------------------------------------------
def set_dict_entry(key_sequence: Iterable[str], data_dict: AnyDict, value: object) -> None:
    try:
        reduce(operator.getitem, key_sequence[:-1], data_dict)[key_sequence[-1]] = value
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e


# --------------------------------------------------------------------------------------------------
def nested_dict_keys(
    d: dict[str, AnyDict], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    for key, value in d.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from nested_dict_keys(value, path)
        else:
            yield path


# ==================================================================================================
@dataclass
class ULACConstructorSettings:
    mesh: pv.PolyData
    feature_tags: dict[str, int]
    path_config: PathConfigDict
    marker_config: MarkerConfigDict
    segmentation_workflow: Iterable[workflow.Step]


class ULACConstructor:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ULACConstructorSettings) -> None:
        self._mesh = settings.mesh
        self._feature_tags = settings.feature_tags
        self._path_config = settings.path_config
        self._marker_config = settings.marker_config
        self._segmentation_workflow = settings.segmentation_workflow
        self._marker_data = create_empty_dict_from_keys(settings.marker_config)
        self._raw_path_data = create_empty_dict_from_keys(settings.path_config)
        self._parameterized_path_data = create_empty_dict_from_keys(settings.path_config)
        self._uac_path_data = create_empty_dict_from_keys(settings.path_config)

    # ----------------------------------------------------------------------------------------------
    def construct_segmentation(self) -> None:
        print("Starting Segmentation")
        print("=====================\n\n")

        for i, step in enumerate(self._segmentation_workflow):
            str_to_print = f"Step {i + 1}/{len(self._segmentation_workflow)}: {step.id}"
            print(str_to_print)
            print("-" * len(str_to_print))
            match step.type:
                case "feature_extraction":
                    self._extract_features(step.apply_to)
                case "marker_extraction":
                    self._extract_markers(step.apply_to)
                case "shortest_path_construction":
                    self._construct_shortest_paths(step.apply_to)
                case "path_parameterization":
                    self._parameterize_paths(step.apply_to)
            print(" ")

    # ----------------------------------------------------------------------------------------------
    def construct_uacs(self) -> None:
        key_sequences = nested_dict_keys(self._path_config)
        for key_sequence in key_sequences:
            path_config = get_dict_entry(key_sequence, self._path_config)
            if path_config.parameterization is None:
                continue
            print(f"Constructing UACs for Path: {key_sequence}")
            parameterized_path = get_dict_entry(key_sequence, self._parameterized_path_data)
            if not isinstance(parameterized_path, base.ParameterizedPath):
                raise TypeError(
                    f"Parameterized path {key_sequence} for UAC construction has not been "
                    "constructed yet."
                )
            path_markers = []
            for marker_config in path_config.parameterization.markers:
                marker = get_dict_entry(marker_config, self._marker_data)
                if not isinstance(marker, base.Marker):
                    raise TypeError(
                        f"Marker {marker_config} for path {key_sequence} has not been constructed "
                        "yet."
                    )
                path_markers.append(marker)
            marker_uacs = [marker.uacs for marker in path_markers]
            marker_values = path_config.parameterization.marker_relative_positions
            if len(path_markers) == 2:
                start_uac, end_uac = marker_uacs
                uac_path = base.compute_uacs_line(parameterized_path, start_uac, end_uac)
            else:
                uac_path = base.compute_uacs_polygon(parameterized_path, marker_values, marker_uacs)
            set_dict_entry(key_sequence, self._uac_path_data, uac_path)

    # ----------------------------------------------------------------------------------------------
    @property
    def data(self) -> tuple[MarkerDict, PathDict, ParameterizedPathDict, UACPathDict]:
        return (
            self._marker_data,
            self._raw_path_data,
            self._parameterized_path_data,
            self._uac_path_data,
        )

    # ----------------------------------------------------------------------------------------------
    @data.setter
    def data(self, data: tuple[MarkerDict, PathDict, ParameterizedPathDict, UACPathDict]) -> None:
        (
            self._marker_data,
            self._raw_path_data,
            self._parameterized_path_data,
            self._uac_path_data,
        ) = data

    # ----------------------------------------------------------------------------------------------
    def _extract_features(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._path_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            try:
                path_config = get_dict_entry(key_sequence, self._path_config)
            except KeyError as e:
                raise KeyError(f"Key sequence {key_sequence} not found in path_config") from e
            if not isinstance(path_config, config.BoundaryPathConfig):
                continue
            print(f"Extracting Feature: {key_sequence}")
            tag_value = self._feature_tags[path_config.feature_tag]
            is_mesh_boundary = path_config.coincides_with_mesh_boundary
            boundary_path = base.get_feature_boundary(self._mesh, tag_value, is_mesh_boundary)
            set_dict_entry(key_sequence, self._raw_path_data, boundary_path)

    # ----------------------------------------------------------------------------------------------
    def _extract_markers(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._marker_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            marker_config = get_dict_entry(key_sequence, self._marker_config)
            print(f"Extracting Marker: {key_sequence}")
            if marker_config.position_type == "index":
                containing_path = get_dict_entry(marker_config.path, self._raw_path_data)
                if not isinstance(containing_path, np.ndarray):
                    raise ValueError(
                        f"Raw path {marker_config.path} for marker "
                        f"{key_sequence} has not been constructed yet."
                    )
                try:
                    marker_ind = containing_path[marker_config.position]
                except IndexError as e:
                    raise IndexError(
                        f"Index {marker_config.position} out of bounds for path "
                        f"{marker_config.path} with length {len(containing_path)}"
                    ) from e
            elif marker_config.position_type == "relative":
                containing_path = get_dict_entry(marker_config.path, self._parameterized_path_data)
                if not isinstance(containing_path, base.ParameterizedPath):
                    raise TypeError(
                        f"Parameterized path {marker_config.path} for marker "
                        f"{key_sequence} has not been constructed yet."
                    )
                try:
                    relative_marker_ind = np.where(
                        containing_path.relative_lengths >= marker_config.position
                    )[0][0]
                    marker_ind = containing_path.inds[relative_marker_ind]
                except IndexError as e:
                    raise ValueError(
                        f"Relative position {marker_config.position} not found on path "
                        f"{marker_config.path}"
                    ) from e
            else:
                raise ValueError(
                    f"Unknown position_type {marker_config.position_type} for marker {key_sequence}"
                )
            marker = base.Marker(ind=marker_ind, uacs=marker_config.uacs)
            set_dict_entry(key_sequence, self._marker_data, marker)

    # ----------------------------------------------------------------------------------------------
    def _construct_shortest_paths(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._path_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            path_config = get_dict_entry(key_sequence, self._path_config)
            if not isinstance(path_config, config.ConnectionPathConfig):
                continue
            print(f"Constructing Shortest Path: {key_sequence}")
            boundaries = []
            for boundary_type, boundary_id in zip(
                path_config.boundary_types, (path_config.start, path_config.end), strict=True
            ):
                if boundary_type == "path":
                    boundary_path = get_dict_entry(boundary_id, self._raw_path_data)
                    if not isinstance(boundary_path, np.ndarray):
                        raise ValueError(
                            f"Raw path {boundary_id} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                elif boundary_type == "marker":
                    boundary_marker = get_dict_entry(boundary_id, self._marker_data)
                    if not isinstance(boundary_marker, base.Marker):
                        raise ValueError(
                            f"Marker {boundary_id} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                    boundary_path = np.array((boundary_marker.ind,), dtype=int)
                else:
                    raise ValueError(
                        f"Unknown boundary type {boundary_type} for path {key_sequence}"
                    )
                boundaries.append(boundary_path)

            if path_config.inadmissible is None:
                inadmissible_set = np.array([], dtype=int)
            else:
                inadmissible_sets = []
                for inadmissible in path_config.inadmissible:
                    inadmissible_path = get_dict_entry(inadmissible, self._raw_path_data)
                    if not isinstance(inadmissible_path, np.ndarray):
                        raise TypeError(
                            f"Raw path {inadmissible} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                    inadmissible_sets.append(inadmissible_path)
                inadmissible_set = np.unique(np.concatenate(inadmissible_sets))
            shortest_path = base.construct_shortest_path_between_subsets(
                self._mesh,
                *boundaries,
                inadmissible_set,
            )
            set_dict_entry(key_sequence, self._raw_path_data, shortest_path)

    # ----------------------------------------------------------------------------------------------
    def _parameterize_paths(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._path_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            path_config = get_dict_entry(key_sequence, self._path_config)
            if path_config.parameterization is None:
                continue
            print(f"Parameterizing Path: {key_sequence}")
            path = get_dict_entry(key_sequence, self._raw_path_data)
            if not isinstance(path, np.ndarray):
                raise TypeError(
                    f"Raw path {key_sequence} for parameterization has not been constructed yet."
                )
            path_markers = []
            for marker_config in path_config.parameterization.markers:
                marker = get_dict_entry(marker_config, self._marker_data)
                if not isinstance(marker, base.Marker):
                    raise TypeError(
                        f"Marker {marker_config} for path {key_sequence} has not been constructed "
                        "yet."
                    )
                path_markers.append(marker)
            marker_inds = [marker.ind for marker in path_markers]
            parameterized_path = base.parameterize_path(
                self._mesh,
                path,
                marker_inds,
                path_config.parameterization.marker_relative_positions,
            )
            set_dict_entry(key_sequence, self._parameterized_path_data, parameterized_path)
