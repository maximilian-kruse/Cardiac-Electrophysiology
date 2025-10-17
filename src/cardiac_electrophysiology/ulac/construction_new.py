import operator
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
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
type SubmeshBoundaryDict = dict[
    str, "SubmeshBoundaryDict" | np.ndarray[tuple[int], np.dtype[np.int64]]
]
type SubmeshDict = dict[str, "SubmeshDict" | pv.PolyData]
type UACSubmeshDict = dict[str, "UACSubmeshDict" | pv.PolyData]

type MarkerConfigDict = dict[str, "MarkerConfigDict" | config.MarkerConfig]
type PathConfigDict = dict[
    str, "PathConfigDict" | config.BoundaryPathConfig | config.ConnectionPathConfig
]
type SubmeshConfigDict = dict[str, "SubmeshConfigDict" | config.SubmeshConfig]
type AnyDict = (
    MarkerDict
    | PathDict
    | ParameterizedPathDict
    | UACPathDict
    | MarkerConfigDict
    | PathConfigDict
    | SubmeshConfigDict
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
    submesh_config: SubmeshConfigDict
    segmentation_workflow: Iterable[workflow.Step]


@dataclass
class ULACData:
    marker_data: MarkerDict | None = None
    raw_path_data: PathDict | None = None
    parameterized_path_data: ParameterizedPathDict | None = None
    uac_path_data: UACPathDict | None = None
    submesh_boundary_data: SubmeshBoundaryDict | None = None
    submesh_data: SubmeshDict | None = None
    uac_submesh_data: UACSubmeshDict | None = None


class ULACConstructor:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ULACConstructorSettings) -> None:
        self._mesh = settings.mesh
        self._feature_tags = settings.feature_tags
        self._path_config = settings.path_config
        self._marker_config = settings.marker_config
        self._submesh_config = settings.submesh_config
        self._segmentation_workflow = settings.segmentation_workflow
        self._marker_data = create_empty_dict_from_keys(settings.marker_config)
        self._raw_path_data = create_empty_dict_from_keys(settings.path_config)
        self._parameterized_path_data = create_empty_dict_from_keys(settings.path_config)
        self._uac_path_data = create_empty_dict_from_keys(settings.path_config)
        self._submesh_boundary_data = create_empty_dict_from_keys(settings.path_config)
        self._submesh_data = create_empty_dict_from_keys(settings.submesh_config)
        self._uac_submesh_data = create_empty_dict_from_keys(settings.submesh_config)

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
        self._construct_uac_paths()
        self._extract_submesh_boundaries()

    # ----------------------------------------------------------------------------------------------
    @property
    def data(self) -> ULACData:
        ulac_data = ULACData(
            marker_data=self._marker_data,
            raw_path_data=self._raw_path_data,
            parameterized_path_data=self._parameterized_path_data,
            uac_path_data=self._uac_path_data,
            submesh_boundary_data=self._submesh_boundary_data,
            submesh_data=self._submesh_data,
            uac_submesh_data=self._uac_submesh_data,
        )
        return ulac_data

    # ----------------------------------------------------------------------------------------------
    @data.setter
    def data(self, data: ULACData) -> None:
        for input_field, class_attr in zip(
            fields(data),
            (
                "_marker_data",
                "_raw_path_data",
                "_parameterized_path_data",
                "_uac_path_data",
                "_submesh_boundary_data",
                "_submesh_data",
                "_uac_submesh_data",
            ),
            strict=True,
        ):
            input_data = getattr(data, input_field.name)
            if input_data is not None:
                setattr(self, class_attr, input_data)

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

            # Get marker from index in raw path
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

            # Get marker from relative position in parameterized path
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

            # Set marker
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

            # Get boundary sets
            for boundary_type, boundary_id in zip(
                path_config.boundary_types, (path_config.start, path_config.end), strict=True
            ):
                # Option 1: Boundary set is a raw path
                if boundary_type == "path":
                    boundary_path = get_dict_entry(boundary_id, self._raw_path_data)
                    if not isinstance(boundary_path, np.ndarray):
                        raise ValueError(
                            f"Raw path {boundary_id} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                # Option 2: Boundary set is a marker
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

            # Get inadmissible sets
            inadmissible_sets = []
            for inadmissible in (path_config.inadmissible_contact, path_config.inadmissible_along):
                if inadmissible is None:
                    subset = np.array([], dtype=int)
                else:
                    subsets = []
                    for subset_id in inadmissible:
                        subset_path = get_dict_entry(subset_id, self._raw_path_data)
                        if not isinstance(subset_path, np.ndarray):
                            raise TypeError(
                                f"Raw path {subset_id} for path {key_sequence} has not been "
                                "constructed yet."
                            )
                        subsets.append(subset_path)
                    subset = np.unique(np.concatenate(subsets))
                inadmissible_sets.append(subset)

            # Compute shortest Path
            shortest_path = base.construct_shortest_path_between_subsets(
                self._mesh,
                *boundaries,
                *inadmissible_sets,
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

            # Get raw path
            path = get_dict_entry(key_sequence, self._raw_path_data)
            if not isinstance(path, np.ndarray):
                raise TypeError(
                    f"Raw path {key_sequence} for parameterization has not been constructed yet."
                )

            # Get marker data
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

            # Parameterize path
            parameterized_path = base.parameterize_path(
                self._mesh,
                path,
                marker_inds,
                path_config.parameterization.marker_relative_positions,
            )
            set_dict_entry(key_sequence, self._parameterized_path_data, parameterized_path)

    # ----------------------------------------------------------------------------------------------
    def _construct_uac_paths(self) -> None:
        key_sequences = nested_dict_keys(self._path_config)
        for key_sequence in key_sequences:
            path_config = get_dict_entry(key_sequence, self._path_config)
            if path_config.parameterization is None:
                continue
            print(f"Constructing UACs for Path: {key_sequence}")

            # Get parameterized path
            parameterized_path = get_dict_entry(key_sequence, self._parameterized_path_data)
            if not isinstance(parameterized_path, base.ParameterizedPath):
                raise TypeError(
                    f"Parameterized path {key_sequence} for UAC construction has not been "
                    "constructed yet."
                )

            # Get marker data
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

            # Compute UACs
            if len(path_markers) == 2:
                start_uac, end_uac = marker_uacs
                uac_path = base.compute_uacs_line(parameterized_path, start_uac, end_uac)
            else:
                uac_path = base.compute_uacs_polygon(parameterized_path, marker_values, marker_uacs)
            set_dict_entry(key_sequence, self._uac_path_data, uac_path)

    # ----------------------------------------------------------------------------------------------
    def _extract_submesh_boundaries(self) -> None:
        key_sequences = nested_dict_keys(self._submesh_config)
        for key_sequence in key_sequences:
            submesh_config = get_dict_entry(key_sequence, self._submesh_config)
            print(f"Extracting boundaries for Submesh: {key_sequence}")

            boundary_inds = []
            boundary_alpha = []
            boundary_beta = []
            for path, portion in zip(
                submesh_config.boundary_paths, submesh_config.portions, strict=True
            ):
                uac_path = get_dict_entry(path, self._uac_path_data)
                if not isinstance(uac_path, base.UACPath):
                    raise TypeError(
                        f"UAC path {uac_path} for submesh {key_sequence} "
                        "has not been constructed yet."
                    )
                relevant_section = np.where(
                    (uac_path.relative_lengths >= portion[0])
                    & (uac_path.relative_lengths <= portion[1])
                )[0]
                boundary_inds.append(uac_path.inds[relevant_section])
                boundary_alpha.append(uac_path.alpha[relevant_section])
                boundary_beta.append(uac_path.beta[relevant_section])

            unique_inds, unique_mask = np.unique(np.concatenate(boundary_inds), return_index=True)
            unique_alpha = np.concatenate(boundary_alpha)[unique_mask]
            unique_beta = np.concatenate(boundary_beta)[unique_mask]
            submesh_boundary = base.UACPath(inds=unique_inds, alpha=unique_alpha, beta=unique_beta)
            set_dict_entry(key_sequence, self._submesh_boundary_data, submesh_boundary)
