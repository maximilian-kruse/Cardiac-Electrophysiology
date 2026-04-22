from pathlib import Path

import basix
import dolfinx
import meshio
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import ufl
from mpi4py import MPI


# ==================================================================================================
def convert_unstructured_to_polydata_mesh(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    point_data = mesh.point_data
    cell_data = mesh.cell_data
    polydata_mesh = pv.PolyData(mesh.points, mesh.cells)
    for key in point_data:
        polydata_mesh.point_data[key] = point_data[key]
    for key in cell_data:
        polydata_mesh.cell_data[key] = cell_data[key]

    return polydata_mesh


# --------------------------------------------------------------------------------------------------
def convert_vtu_to_xdmf_mesh(input_path: Path) -> None:
    output_path = input_path.parent / "mesh.xdmf"
    mesh = meshio.read(str(input_path))
    meshio.write(str(output_path), mesh, file_format="xdmf")


# --------------------------------------------------------------------------------------------------
def create_dolfinx_mesh_from_pyvista_mesh(pv_mesh: pv.UnstructuredGrid) -> dolfinx.mesh.Mesh:
    points = pv_mesh.points
    cells = pv_mesh.cells.reshape(-1, 4)[:, 1:]
    if not np.all(pv_mesh.celltypes == pv.CellType.TRIANGLE):
        raise ValueError("Only triangular meshes are supported.")
    ufl_type = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, ufl_type)


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
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> sp.coo_array:
    pv_mesh = pv.PolyData.from_regular_faces(vertex_coordinates, connectivity)
    simplex_areas = np.asarray(pv_mesh.compute_cell_sizes().cell_data["Area"], dtype=np.float64)

    num_vertices = np.max(connectivity) + 1
    num_simplices = connectivity.shape[0]
    row_inds = connectivity.flatten()
    col_inds = np.repeat(np.arange(num_simplices), 3)
    data = np.repeat(simplex_areas, 3)
    weighted_adjacency = sp.coo_array(
        (data, (row_inds, col_inds)), shape=(num_vertices, num_simplices)
    )
    area_sum_per_vertex = np.asarray(weighted_adjacency.sum(axis=1)).ravel()
    inv_area_sum_per_vertex = 1 / area_sum_per_vertex
    normalization_matrix = sp.diags_array(inv_area_sum_per_vertex).tocoo()
    simplex_to_vertex_matrix = normalization_matrix @ weighted_adjacency
    return simplex_to_vertex_matrix
