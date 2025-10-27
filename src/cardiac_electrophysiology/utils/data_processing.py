import numpy as np
import scipy.sparse as sp
import scipy.stats as st


# ==================================================================================================
def compute_axial_mean_and_variance(
    angle_samples: np.ndarray[tuple[int], np.dtype[np.float64]], axis: int = 0
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
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
def assemble_vertex_to_simplex_interpolation_matrix(
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> sp.coo_array:
    num_vertices = np.max(connectivity) + 1
    num_simplices = connectivity.shape[0]
    row_inds = np.repeat(np.arange(num_simplices), 3)
    col_inds = connectivity.flatten()
    data = np.full(num_simplices * 3, 1 / 3)
    interpolation_matrix = sp.coo_array(
        (data, (row_inds, col_inds)), shape=(num_simplices, num_vertices)
    )
    return interpolation_matrix


# --------------------------------------------------------------------------------------------------
def assemble_simplex_to_vertex_interpolation_matrix(
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> sp.coo_array:
    vertex_to_simplex_matrix = assemble_vertex_to_simplex_interpolation_matrix(connectivity)
    num_adjacent_cells = vertex_to_simplex_matrix.sum(axis=0)
    normalization_matrix = sp.diags_array(1 / num_adjacent_cells).tocoo()
    simplex_to_vertex_matrix = normalization_matrix @ vertex_to_simplex_matrix.transpose()
    return simplex_to_vertex_matrix
