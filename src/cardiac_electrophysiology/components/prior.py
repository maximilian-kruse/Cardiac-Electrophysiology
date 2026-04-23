from typing import override

import dolfinx
import numpy as np
from ls_prior import builder as ls_prior_builder

from cardiac_electrophysiology.ls_bip import components as ls_bip_components


# ==================================================================================================
class DolfinxMeshMapping:
    def __init__(self, fenicsx_mesh: dolfinx.mesh.Mesh) -> None:
        self._fx_original_vertex_inds = fenicsx_mesh.geometry.input_global_indices

    def map_vertex_data_to_dolfinx_ordering(
        self, data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return data_array[self._fx_original_vertex_inds]

    def map_vertex_data_from_dolfinx_ordering(
        self, dlx_data_array: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        data_array = np.empty_like(dlx_data_array)
        data_array[self._fx_original_vertex_inds] = dlx_data_array
        return data_array


# ==================================================================================================
class AngleFieldPrior(ls_bip_components.Prior):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ls_prior_builder.BilaplacianPriorSettings) -> None:
        self._dlx_mesh_mapping = DolfinxMeshMapping(settings.mesh)
        settings.mean_vector = self._dlx_mesh_mapping.map_vertex_data_to_dolfinx_ordering(
            settings.mean_vector
        )
        prior_builder = ls_prior_builder.BilaplacianPriorBuilder(settings)
        self._spde_prior = prior_builder.build()

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        dlx_parameter_vector = self._dlx_mesh_mapping.map_vertex_data_to_dolfinx_ordering(
            parameter_vector
        )
        cost = self._spde_prior.evaluate_cost(dlx_parameter_vector)
        return cost

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_parameter_vector = self._dlx_mesh_mapping.map_vertex_data_to_dolfinx_ordering(
            parameter_vector
        )
        dlx_gradient = self._spde_prior.evaluate_gradient(dlx_parameter_vector)
        gradient = self._dlx_mesh_mapping.map_vertex_data_from_dolfinx_ordering(dlx_gradient)
        return gradient

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_direction_vector = self._dlx_mesh_mapping.map_vertex_data_to_dolfinx_ordering(
            direction_vector
        )
        dlx_hessian_vector_product = self._spde_prior.evaluate_hessian_vector_product(
            dlx_direction_vector
        )
        hessian_vector_product = self._dlx_mesh_mapping.map_vertex_data_from_dolfinx_ordering(
            dlx_hessian_vector_product
        )
        return hessian_vector_product

    # ----------------------------------------------------------------------------------------------
    @override
    def generate_sample(
        self, _parameter_vector: None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        dlx_sample = self._spde_prior.generate_sample()
        sample = self._dlx_mesh_mapping.map_vertex_data_from_dolfinx_ordering(dlx_sample)
        return sample
