from dataclasses import dataclass

import numpy as np
import pyvista as pv
from eikonax import preprocessing as ekx_preprocessing
from eikonax import solver as ekx_solver
from eikonax import tensorfield as ekx_tensorfield

from . import preprocessing


# ==================================================================================================
@dataclass
class UnOrderedPath:
    points: np.ndarray=None
    edges: np.ndarray=None


# --------------------------------------------------------------------------------------------------
def get_boundary_point_coordinates(region: pv.PolyData) -> pv.ArrayLike:
    region_boundaries = region.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    region_boundary_points = region_boundaries.points.view(
        [("", region_boundaries.points.dtype)] * region_boundaries.points.shape[1]
    )

    return region_boundary_points


# --------------------------------------------------------------------------------------------------
def get_inner_outer_boundaries(
    mesh: pv.PolyData, feature_tag: int, mesh_boundary_points: np.ndarray
) -> tuple[UnOrderedPath, UnOrderedPath]:
    feature_region = mesh.extract_values(feature_tag, scalars="anatomical_tags")
    feature_boundaries = feature_region.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    feature_boundary_points = get_boundary_point_coordinates(feature_region)
    is_inner_boundary = np.where(~np.isin(feature_boundary_points, mesh_boundary_points))[0]
    is_outer_boundary = np.where(np.isin(feature_boundary_points, mesh_boundary_points))[0]
    inner_boundary_inds = feature_region.point_data["vtkOriginalPointIds"][
        feature_boundaries.point_data["vtkOriginalPointIds"][is_inner_boundary]
    ]
    outer_boundary_inds = feature_region.point_data["vtkOriginalPointIds"][
        feature_boundaries.point_data["vtkOriginalPointIds"][is_outer_boundary]
    ]
    inner_boundary_tags = np.zeros(feature_boundaries.number_of_points)
    inner_boundary_tags[is_inner_boundary] = 1
    outer_boundary_tags = np.zeros(feature_boundaries.number_of_points)
    outer_boundary_tags[is_outer_boundary] = 1
    feature_boundaries.point_data["inner_boundary"] = inner_boundary_tags
    feature_boundaries.point_data["outer_boundary"] = outer_boundary_tags
    inner_boundary_feature = feature_boundaries.extract_values(1, scalars="inner_boundary")
    outer_boundary_feature = feature_boundaries.extract_values(1, scalars="outer_boundary")
    inner_boundary_connectivity = inner_boundary_inds[
        inner_boundary_feature.cells.reshape(-1, 3)[:, 1:]
    ]
    outer_boundary_connectivity = outer_boundary_inds[
        outer_boundary_feature.cells.reshape(-1, 3)[:, 1:]
    ]
    inner_boundary_path = UnOrderedPath(
        np.array(inner_boundary_inds, dtype=int), np.array(inner_boundary_connectivity)
    )
    outer_boundary_path = UnOrderedPath(
        np.array(outer_boundary_inds, dtype=int), np.array(outer_boundary_connectivity)
    )

    return inner_boundary_path, outer_boundary_path


# --------------------------------------------------------------------------------------------------
def construct_path_from_boundary(boundary: UnOrderedPath) -> np.ndarray:
    start_point = boundary.points[0]
    adjacent_edges = boundary.edges[np.where(boundary.edges == start_point)[0][0]]
    end_point = adjacent_edges[adjacent_edges != start_point][0]

    ordered_points = [start_point]
    current_point = start_point
    last_point = end_point

    while current_point != end_point:
        adjacent_edges = np.where(boundary.edges == current_point)[0]
        edge_candidate = boundary.edges[adjacent_edges[0]]
        if last_point is None or last_point not in edge_candidate:
            new_point = edge_candidate[edge_candidate != current_point][0]
        else:
            edge_candidate = boundary.edges[adjacent_edges[1]]
            new_point = edge_candidate[edge_candidate != current_point][0]
        ordered_points.append(int(new_point))
        last_point = current_point
        current_point = new_point

    return np.array(ordered_points)


# --------------------------------------------------------------------------------------------------
def prepare_eikonal_run(input_mesh: pv.PolyData) -> tuple:
    vertices = input_mesh.points
    simplices = preprocessing.convert_pv_cells_to_numpy_cells(input_mesh.faces)
    mesh_data = ekx_preprocessing.MeshData(vertices, simplices)
    solver_data = ekx_solver.SolverData(
        tolerance=1e-6,
        max_num_iterations=1000,
        max_value=100000,
        loop_type="jitted_while",
        use_soft_update=False,
        softminmax_order=100,
        softminmax_cutoff=0.01,
    )
    parameter_vector = np.ones(simplices.shape[0])

    tensor_on_simplex = ekx_tensorfield.InvLinearScalarSimplexTensor(dimension=vertices.shape[1])
    tensor_field_mapping = ekx_tensorfield.LinearScalarMap()
    tensor_field_object = ekx_tensorfield.TensorField(
        num_simplices=simplices.shape[0],
        vector_to_simplices_map=tensor_field_mapping,
        simplex_tensor=tensor_on_simplex,
    )
    tensor_field_instance = tensor_field_object.assemble_field(parameter_vector)
    return mesh_data, solver_data, tensor_field_instance


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_boundaries(
    mesh: pv.PolyData, boundary_one: np.ndarray, boundary_two: np.ndarray, solver_setup: tuple
) -> np.ndarray:
    optimal_boundary_one_ind, optimal_boundary_two_ind = (
        _get_shortest_path_points_between_boundaries(boundary_one, boundary_two, solver_setup)
        )
    shortest_path = _construct_shortest_path_from_end_points(
        mesh, optimal_boundary_one_ind, optimal_boundary_two_ind
    )
    return shortest_path


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_boundary_and_point(
    mesh: pv.PolyData, boundary: np.ndarray, point: int, solver_setup: tuple
) -> np.ndarray:
    optimal_boundary_ind = _get_shortest_path_points_between_boundary_and_point(
        boundary, point, solver_setup
    )
    shortest_path = _construct_shortest_path_from_end_points(mesh, optimal_boundary_ind, point)
    return shortest_path


# ==================================================================================================
def _get_shortest_path_points_between_boundaries(
    boundary_one: np.ndarray, boundary_two: np.ndarray, solver_setup: tuple
) -> np.ndarray:
    mesh_data, solver_data, tensor_field_instance = solver_setup
    initial_sites = ekx_preprocessing.InitialSites(
        inds=boundary_one,
        values=np.zeros_like(boundary_one),
    )
    eikonal_solver = ekx_solver.Solver(mesh_data, solver_data, initial_sites)
    solution = eikonal_solver.run(tensor_field_instance)
    solution_on_boundary_two = solution.values[boundary_two]

    optimal_boundary_two_ind = boundary_two[int(np.argmin(solution_on_boundary_two))]
    optimal_boundary_one_ind = _get_shortest_path_points_between_boundary_and_point(
        boundary_one, optimal_boundary_two_ind, solver_setup
    )

    return optimal_boundary_one_ind, optimal_boundary_two_ind


# --------------------------------------------------------------------------------------------------
def _get_shortest_path_points_between_boundary_and_point(
    boundary: np.ndarray, point: int, solver_setup: tuple
) -> np.ndarray:
    mesh_data, solver_data, tensor_field_instance = solver_setup
    initial_sites = ekx_preprocessing.InitialSites(
        inds=(point,),
        values=(0,),
    )
    eikonal_solver = ekx_solver.Solver(mesh_data, solver_data, initial_sites)
    solution = eikonal_solver.run(tensor_field_instance)
    solution_on_boundary_one = solution.values[boundary]
    optimal_boundary_ind = boundary[int(np.argmin(solution_on_boundary_one))]

    return optimal_boundary_ind


# --------------------------------------------------------------------------------------------------
def _construct_shortest_path_from_end_points(
    mesh: pv.PolyData, start_point: int, end_point: int
) -> np.ndarray:
    shortest_path = mesh.geodesic(start_point, end_point).point_data["vtkOriginalPointIds"]
    return np.array(shortest_path)
