import numpy as np
import preprocess_mesh as pm
import pyvista as pv
from eikonax import preprocessing as ekx_preprocessing
from eikonax import solver as ekx_solver
from eikonax import tensorfield as ekx_tensorfield


# ==================================================================================================
def get_feature_inner_boundaries(input_mesh: pv.PolyData, feature_tags: list[int]):
    mesh_boundaries = input_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    mesh_boundary_points = mesh_boundaries.points.view(
        [("", mesh_boundaries.points.dtype)] * mesh_boundaries.points.shape[1]
    )

    inner_boundaries = []
    inner_boundary_global_inds = []
    for feature in feature_tags:
        inner_boundary_inds, inner_boundary = _get_feature_inner_boundary(
            input_mesh, feature, mesh_boundary_points
        )
        inner_boundaries.append(inner_boundary)
        inner_boundary_global_inds.append(inner_boundary_inds)

    return inner_boundaries, inner_boundary_global_inds


def compute_roof_frame(
    input_mesh: pv.PolyData, anatomical_tags: dict, inner_boundary_global_inds: list[np.ndarray]
) -> list[pv.PolyData]:
    solver_setup = prepare_eikonal_run(input_mesh)

    lipv_boundary = inner_boundary_global_inds[anatomical_tags["LIPV"] - 1]
    lspv_boundary = inner_boundary_global_inds[anatomical_tags["LSPV"] - 1]
    lipv_point, lspv_point = find_shortest_path_between_boundaries(
        lipv_boundary, lspv_boundary, solver_setup
    )
    lipv_lspv_path = input_mesh.geodesic(lipv_point, lspv_point)

    ripv_boundary = inner_boundary_global_inds[anatomical_tags["RIPV"] - 1]
    rspv_boundary = inner_boundary_global_inds[anatomical_tags["RSPV"] - 1]
    ripv_point, rspv_point = find_shortest_path_between_boundaries(
        ripv_boundary, rspv_boundary, solver_setup
    )
    ripv_rspv_path = input_mesh.geodesic(ripv_point, rspv_point)

    lipv_boundary = inner_boundary_global_inds[anatomical_tags["LIPV"] - 1]
    ripv_boundary = inner_boundary_global_inds[anatomical_tags["RIPV"] - 1]
    lipv_point, ripv_point = find_shortest_path_between_boundaries(
        lipv_boundary, ripv_boundary, solver_setup
    )
    lipv_ripv_path = input_mesh.geodesic(lipv_point, ripv_point)

    lspv_boundary = inner_boundary_global_inds[anatomical_tags["LSPV"] - 1]
    rspv_boundary = inner_boundary_global_inds[anatomical_tags["RSPV"] - 1]
    lspv_point, rspv_point = find_shortest_path_between_boundaries(
        lspv_boundary, rspv_boundary, solver_setup
    )
    lspv_rspv_path = input_mesh.geodesic(lspv_point, rspv_point)
    return lipv_lspv_path, ripv_rspv_path, lipv_ripv_path, lspv_rspv_path


# --------------------------------------------------------------------------------------------------
def prepare_eikonal_run(input_mesh):
    vertices = input_mesh.points
    simplices = pm.convert_pv_cells_to_numpy_cells(input_mesh.faces)
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
def find_shortest_path_between_boundaries(
    boundary_one_inds: np.ndarray, boundary_two_inds: np.ndarray, solver_setup: tuple
) -> tuple[int, int]:
    mesh_data, solver_data, tensor_field_instance = solver_setup
    initial_sites = ekx_preprocessing.InitialSites(
        inds=boundary_one_inds,
        values=np.zeros_like(boundary_one_inds),
    )
    eikonal_solver = ekx_solver.Solver(mesh_data, solver_data, initial_sites)
    solution = eikonal_solver.run(tensor_field_instance)
    solution_on_boundary_two = solution.values[boundary_two_inds]

    optimal_boundary_two_ind = boundary_two_inds[int(np.argmin(solution_on_boundary_two))]

    initial_sites = ekx_preprocessing.InitialSites(
        inds=(optimal_boundary_two_ind,),
        values=(0,),
    )
    eikonal_solver = ekx_solver.Solver(mesh_data, solver_data, initial_sites)
    solution = eikonal_solver.run(tensor_field_instance)
    solution_on_boundary_one = solution.values[boundary_one_inds]
    optimal_boundary_one_ind = boundary_one_inds[int(np.argmin(solution_on_boundary_one))]

    return optimal_boundary_one_ind, optimal_boundary_two_ind


# --------------------------------------------------------------------------------------------------
def construct_path_between_boundary_points(
    boundary: pv.PolyData, start_point: int, end_point: int, direction: tuple[int, int] = (0, 1)
) -> list[int]:
    connectivity = boundary.cells.reshape(-1, 3)[:, 1:]
    ordered_nodes = [start_point]
    current_node = start_point
    last_node = None

    while current_node != end_point:
        adjacent_segments = np.where(connectivity == current_node)[0]
        segment_candidate = connectivity[adjacent_segments[direction[0]]]
        if last_node is None or last_node not in segment_candidate:
            new_node = segment_candidate[segment_candidate != current_node][0]
        else:
            segment_candidate = connectivity[adjacent_segments[direction[1]]]
            new_node = segment_candidate[segment_candidate != current_node][0]
        ordered_nodes.append(new_node)
        last_node = current_node
        current_node = new_node

    return np.array(ordered_nodes)


# ==================================================================================================
def _get_feature_inner_boundary(
    input_mesh: pv.PolyData, feature_tag: int, mesh_boundary_points: pv.ArrayLike
) -> tuple[np.ndarray, pv.PolyData]:
    feature_region = input_mesh.extract_values(feature_tag, scalars="anatomical_tags")
    feature_boundaries = feature_region.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    feature_boundary_points = feature_boundaries.points.view(
        [("", feature_boundaries.points.dtype)] * feature_boundaries.points.shape[1]
    )
    inner_boundary = np.where(~np.isin(feature_boundary_points, mesh_boundary_points))[0]
    inner_boundary_global_inds = feature_region.point_data["vtkOriginalPointIds"][
        feature_boundaries.point_data["vtkOriginalPointIds"][inner_boundary]
    ]

    inner_boundary_tags = np.zeros(feature_boundaries.number_of_points)
    inner_boundary_tags[inner_boundary] = 1
    feature_boundaries.point_data["inner_boundary"] = inner_boundary_tags
    inner_boundary_feature = feature_boundaries.extract_values(1, scalars="inner_boundary")
    inner_boundary_feature.clear_data()

    return inner_boundary_global_inds, inner_boundary_feature
