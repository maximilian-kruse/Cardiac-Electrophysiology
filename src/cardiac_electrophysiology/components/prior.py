from typing import override

import numpy as np
from ls_prior import builder as ls_prior_builder

from cardiac_electrophysiology.ls_bip import components as ls_bip_components
from cardiac_electrophysiology.utils import data_processing, mesh_processing


# ==================================================================================================
class FiberFieldPrior(ls_bip_components.Prior):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ls_prior_builder.BilaplacianPriorSettings) -> None:
        dlx_mesh = settings.mesh
        vertex_coordinates = dlx_mesh.geometry.x
        connectivity = dlx_mesh.geometry.dofmap
        self._dlx_mesh_mapping = mesh_processing.DolfinxMeshMapping(dlx_mesh)
        self._vertex_to_simplex_matrix = (
            data_processing.assemble_vertex_to_simplex_interpolation_matrix(connectivity)
        )
        self._simplex_to_vertex_matrix = (
            data_processing.assemble_simplex_to_vertex_interpolation_matrix(
                connectivity, vertex_coordinates
            )
        )

        settings.mean_vector = (
            self._simplex_to_vertex_matrix
            @ self._dlx_mesh_mapping.map_cell_data_to_dolfinx_ordering(settings.mean_vector)
        )
        prior_builder = ls_prior_builder.BilaplacianPriorBuilder(settings)
        self._spde_prior = prior_builder.build()

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        dlx_parameter_vector_on_cells = self._dlx_mesh_mapping.map_cell_data_to_dolfinx_ordering(
            parameter_vector
        )
        dlx_parameter_vector_on_vertices = (
            self._simplex_to_vertex_matrix @ dlx_parameter_vector_on_cells
        )
        cost = self._spde_prior.evaluate_cost(dlx_parameter_vector_on_vertices)
        return cost

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_parameter_vector_on_cells = self._dlx_mesh_mapping.map_cell_data_to_dolfinx_ordering(
            parameter_vector
        )
        dlx_parameter_vector_on_vertices = (
            self._simplex_to_vertex_matrix @ dlx_parameter_vector_on_cells
        )
        dlx_gradient_on_vertices = self._spde_prior.evaluate_gradient(
            dlx_parameter_vector_on_vertices
        )
        dlx_gradient_on_cells = self._vertex_to_simplex_matrix @ dlx_gradient_on_vertices
        gradient = self._dlx_mesh_mapping.map_cell_data_from_dolfinx_ordering(dlx_gradient_on_cells)
        return gradient

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        _parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @override
    def generate_sample(
        self, _parameter_vector: None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_sample_on_vertices = self._spde_prior.generate_sample()
        dlx_sample_on_cells = self._vertex_to_simplex_matrix @ dlx_sample_on_vertices
        sample = self._dlx_mesh_mapping.map_cell_data_from_dolfinx_ordering(dlx_sample_on_cells)
        return sample
