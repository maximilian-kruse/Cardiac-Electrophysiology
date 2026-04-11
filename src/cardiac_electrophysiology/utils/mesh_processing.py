from pathlib import Path

import dolfinx
import meshio
import numpy as np
import pyvista as pv


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


# ==================================================================================================
def convert_vtu_to_xdmf_mesh(input_path: Path) -> None:
    output_path = input_path.with_suffix(".xdmf")
    mesh = meshio.read(str(input_path))
    meshio.write(str(output_path), mesh, file_format="xdmf")


# ==================================================================================================
class DolfinxMeshMapping:
    def __init__(self, fenicsx_mesh: dolfinx.mesh.Mesh) -> None:
        self._fx_original_vertex_inds = fenicsx_mesh.geometry.input_global_indices
        self._fx_original_cell_inds = fenicsx_mesh.topology.original_cell_index

    def map_vertex_data_to_dolfinx_ordering(
        self, data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return data_array[self._fx_original_vertex_inds]

    def map_cell_data_to_dolfinx_ordering(
        self, data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_data_array = data_array[self._fx_original_cell_inds]
        return dlx_data_array

    def map_vertex_data_from_dolfinx_ordering(
        self, dlx_data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        data_array = np.empty_like(dlx_data_array)
        data_array[self._fx_original_vertex_inds] = dlx_data_array
        return data_array

    def map_cell_data_from_dolfinx_ordering(
        self, dlx_data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        data_array = np.empty_like(dlx_data_array)
        data_array[self._fx_original_cell_inds] = dlx_data_array
        return data_array
