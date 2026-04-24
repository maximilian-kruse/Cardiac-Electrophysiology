from dataclasses import dataclass

import numpy as np
import scipy.stats as st

from cardiac_electrophysiology import posterior_builder
from cardiac_electrophysiology.ls_bip import posterior as lsbip_posterior


# ==================================================================================================
def compute_axial_mean_and_variance(
    angle_samples: np.ndarray[tuple[int, int], np.dtype[np.float64]], axis: int = 1
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    angle_samples = np.atleast_2d(angle_samples)
    normalized_angle_samples = np.mod(angle_samples, np.pi)
    circ_mean_doubled = st.circmean(2 * normalized_angle_samples, axis=axis)
    circ_std_doubled = st.circstd(2 * normalized_angle_samples, axis=axis)
    axial_mean = circ_mean_doubled / 2
    axial_variance = (circ_std_doubled / 2) ** 2

    shift_by_pi_mask = (axial_mean > 1 / 2 * np.pi) & (axial_mean <= 3 / 2 * np.pi)
    shift_by_two_pi_mask = axial_mean > 3 / 2 * np.pi
    axial_mean[shift_by_pi_mask] -= np.pi
    axial_mean[shift_by_two_pi_mask] -= 2 * np.pi

    return axial_mean, axial_variance


# ==================================================================================================
def shift_angles_to_minimize_axial_variance(
    angle_samples: np.ndarray[tuple[int, int], np.dtype[np.float64]], axis: int = 1
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    angle_samples = np.atleast_2d(angle_samples)
    axial_mean, _ = compute_axial_mean_and_variance(angle_samples, axis=axis)
    centered_angles = angle_samples - axial_mean
    wrapped_angles = (centered_angles + np.pi / 2) % np.pi - np.pi / 2
    reconstructed_angles = wrapped_angles + axial_mean
    return reconstructed_angles


# ==================================================================================================
@dataclass
class MapAnalysisData:
    diff_angle_truth_prior: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_angle_truth_map: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_lat_truth_prior: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_lat_truth_map: np.ndarray[tuple[int], np.dtype[np.float64]]
    diff_lat_data_map: np.ndarray[tuple[int], np.dtype[np.float64]]
    ground_truth_parameter: np.ndarray[tuple[int], np.dtype[np.float64]]
    prior_mean_parameter: np.ndarray[tuple[int], np.dtype[np.float64]]
    map_parameter: np.ndarray[tuple[int], np.dtype[np.float64]]


# --------------------------------------------------------------------------------------------------
def compute_map_result_analysis(
    map_parameter: np.ndarray[tuple[int], np.dtype[np.float64]],
    posterior: lsbip_posterior.LogPosterior,
    additional_output: posterior_builder.PosteriorBuilderOutput,
) -> MapAnalysisData:
    prior_mean_parameter = shift_angles_to_minimize_axial_variance(
        additional_output.prior_mean_parameter
    ).flatten()
    ground_truth_parameter = shift_angles_to_minimize_axial_variance(
        additional_output.ground_truth_parameter
    ).flatten()
    map_parameter = shift_angles_to_minimize_axial_variance(map_parameter).flatten()
    prior_mean_predictive = posterior.parameter_to_solution_map.evaluate_forward(
        prior_mean_parameter
    )
    ground_truth_predictive = posterior.parameter_to_solution_map.evaluate_forward(
        ground_truth_parameter
    )
    map_predictive = posterior.parameter_to_solution_map.evaluate_forward(map_parameter)

    diff_angle_truth_prior = compute_axial_data_diff(ground_truth_parameter, prior_mean_parameter)
    diff_angle_truth_map = compute_axial_data_diff(ground_truth_parameter, map_parameter)
    diff_lat_truth_prior = ground_truth_predictive - prior_mean_predictive
    diff_lat_truth_map = ground_truth_predictive - map_predictive
    diff_lat_data_map = (
        additional_output.noisy_data - map_predictive[additional_output.observation_inds]
    )

    print(f"Prior mean angle L2-error: {np.linalg.norm(diff_angle_truth_prior)}")
    print(f"Prior mean angle max-error: {np.max(np.abs(diff_angle_truth_prior))}")
    print(f"MAP angle L2-error: {np.linalg.norm(diff_angle_truth_map)}")
    print(f"MAP angle max-error: {np.max(np.abs(diff_angle_truth_map))}")
    print(f"Prior mean predictive L2-error: {np.linalg.norm(diff_lat_truth_prior)}")
    print(f"Prior mean predictive max-error: {np.max(np.abs(diff_lat_truth_prior))}")
    print(f"MAP predictive L2-error: {np.linalg.norm(diff_lat_truth_map)}")
    print(f"MAP predictive max-error: {np.max(np.abs(diff_lat_truth_map))}")
    print(f"Data predictive L2-error: {np.linalg.norm(diff_lat_data_map)}")
    print(f"Data predictive max-error: {np.max(np.abs(diff_lat_data_map))}")

    return MapAnalysisData(
        diff_angle_truth_prior=diff_angle_truth_prior,
        diff_angle_truth_map=diff_angle_truth_map,
        diff_lat_truth_prior=diff_lat_truth_prior,
        diff_lat_truth_map=diff_lat_truth_map,
        diff_lat_data_map=diff_lat_data_map,
        prior_mean_parameter=prior_mean_parameter,
        ground_truth_parameter=ground_truth_parameter,
        map_parameter=map_parameter,
    )


def compute_axial_data_diff(
    angle_field_one: np.ndarray[tuple[int], np.dtype[np.float64]],
    angle_field_two: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    raw_diff = angle_field_one - angle_field_two
    diff_metric = 0.5 * np.atan2(np.sin(2 * raw_diff), np.cos(2 * raw_diff))
    return diff_metric
