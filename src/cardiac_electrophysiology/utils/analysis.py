from dataclasses import dataclass

import numpy as np

from cardiac_electrophysiology import posterior_builder
from cardiac_electrophysiology.ls_bip import posterior as lsbip_posterior


# ==================================================================================================
@dataclass
class MapAnalysisData:
    diff_angle_truth_prior: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_angle_truth_map: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_lat_truth_prior: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_lat_truth_map: np.ndarray[tuple[int], np.dtype[np.float64]]
    angle_ground_truth: np.ndarray[tuple[int], np.dtype[np.float64]]
    angle_prior_mean: np.ndarray[tuple[int], np.dtype[np.float64]]
    angle_map: np.ndarray[tuple[int], np.dtype[np.float64]]


# --------------------------------------------------------------------------------------------------
def compute_map_result_analysis(
    map_parameter: np.ndarray[tuple[int], np.dtype[np.float64]],
    posterior: lsbip_posterior.LogPosterior,
    additional_output: posterior_builder.PosteriorBuilderOutput,
) -> MapAnalysisData:
    ground_truth_parameter = additional_output.ground_truth_parameter
    mean_parameter = np.zeros_like(ground_truth_parameter)

    map_angles = additional_output.angle_transformator.compute_angle_from_parameter(map_parameter)
    ground_truth_angles = additional_output.angle_transformator.compute_angle_from_parameter(
        ground_truth_parameter
    )
    mean_angles = additional_output.angle_transformator.compute_angle_from_parameter(mean_parameter)

    map_predictive = posterior.parameter_to_solution_map.evaluate_forward(map_parameter)
    ground_truth_predictive = posterior.parameter_to_solution_map.evaluate_forward(
        additional_output.ground_truth_parameter
    )
    mean_predictive = posterior.parameter_to_solution_map.evaluate_forward(mean_parameter)

    diff_angle_truth_prior = compute_axial_data_diff(ground_truth_angles, mean_angles)
    diff_angle_truth_map = compute_axial_data_diff(ground_truth_angles, map_angles)
    diff_lat_truth_prior = ground_truth_predictive - mean_predictive
    diff_lat_truth_map = ground_truth_predictive - map_predictive

    print(f"Prior mean angle L2-error: {np.linalg.norm(diff_angle_truth_prior)}")
    print(f"Prior mean angle max-error: {np.max(np.abs(diff_angle_truth_prior))}")
    print(f"MAP angle L2-error: {np.linalg.norm(diff_angle_truth_map)}")
    print(f"MAP angle max-error: {np.max(np.abs(diff_angle_truth_map))}")
    print(f"Prior mean predictive L2-error: {np.linalg.norm(diff_lat_truth_prior)}")
    print(f"Prior mean predictive max-error: {np.max(np.abs(diff_lat_truth_prior))}")
    print(f"MAP predictive L2-error: {np.linalg.norm(diff_lat_truth_map)}")
    print(f"MAP predictive max-error: {np.max(np.abs(diff_lat_truth_map))}")

    return MapAnalysisData(
        diff_angle_truth_prior=diff_angle_truth_prior,
        diff_angle_truth_map=diff_angle_truth_map,
        diff_lat_truth_prior=diff_lat_truth_prior,
        diff_lat_truth_map=diff_lat_truth_map,
        angle_ground_truth=ground_truth_angles,
        angle_prior_mean=mean_angles,
        angle_map=map_angles,
    )


def compute_axial_data_diff(angle_field_one, angle_field_two):
    raw_diff = angle_field_one - angle_field_two
    diff_field = np.minimum(np.abs(raw_diff), np.pi - np.abs(raw_diff))
    return diff_field
