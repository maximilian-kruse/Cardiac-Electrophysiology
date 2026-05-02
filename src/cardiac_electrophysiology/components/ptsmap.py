from typing import override

import numpy as np
import pyvista as pv
import scipy.sparse as sps
from eikonax import derivator as eikonax_derivator
from eikonax import linalg as eikonax_linalg
from eikonax import solver as eikonax_solver
from eikonax import tensorfield as eikonax_tensorfield

from cardiac_electrophysiology.ls_bip import components as ls_bip_components
from cardiac_electrophysiology.utils import mesh_utils


# ==================================================================================================
class CachedState:
    # ----------------------------------------------------------------------------------------------
    def __init__(self):
        self._parameter_vector = None
        self._solution_vector = None
        self._derivative_solver = None
        self._sparse_partial_parameter = None

        # ----------------------------------------------------------------------------------------------

    def get_parameter_vector(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._parameter_vector

    # ----------------------------------------------------------------------------------------------
    def set_parameter_vector(self, value: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        self._parameter_vector = value
        self._derivative_solver = None
        self._sparse_partial_parameter = None

    # ----------------------------------------------------------------------------------------------
    def get_derivative_solver(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> eikonax_derivator.DerivativeSolver:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            return None
        else:
            return self._derivative_solver

    # ----------------------------------------------------------------------------------------------
    def set_derivative_solver(
        self,
        value: eikonax_derivator.DerivativeSolver,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._derivative_solver = value

    # ----------------------------------------------------------------------------------------------
    def get_sparse_partial_parameter(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> sps.coo_matrix:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            return None
        else:
            return self._sparse_partial_parameter

    # ----------------------------------------------------------------------------------------------
    def set_sparse_partial_parameter(
        self,
        value: sps.coo_matrix,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._sparse_partial_parameter = value


# ==================================================================================================
class EikonalPTSMap(ls_bip_components.ParameterToSolutionMap):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mesh: pv.UnstructuredGrid,
        eikonax_solver: eikonax_solver.Solver,
        eikonax_derivatior: eikonax_derivator.PartialDerivator,
        tensor_field: eikonax_tensorfield.TensorField,
    ) -> None:
        self._eikonax_solver = eikonax_solver
        self._eikonax_derivatior = eikonax_derivatior
        self._tensor_field = tensor_field
        self._vertex_to_simplex_matrix = mesh_utils.assemble_vertex_to_simplex_interpolation_matrix(
            mesh.cells.reshape(-1, 4)[:, 1:]
        )
        self._cached_state = CachedState()

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_forward(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        parameter_vector_on_cells = self._vertex_to_simplex_matrix @ parameter_vector
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector_on_cells)
        solution = self._eikonax_solver.run(tensor_field_instance)
        solution_vector = np.array(solution.values)
        return solution_vector

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        derivative_solver = self._cached_state.get_derivative_solver(parameter_vector)
        sparse_partial_parameter = self._cached_state.get_sparse_partial_parameter(parameter_vector)
        if derivative_solver is None or sparse_partial_parameter is None:
            derivative_solver, sparse_partial_parameter = self._set_up_derivative_structures(
                solution_vector, parameter_vector
            )
            self._cached_state.set_parameter_vector(parameter_vector)
            self._cached_state.set_derivative_solver(derivative_solver, parameter_vector)
            self._cached_state.set_sparse_partial_parameter(
                sparse_partial_parameter, parameter_vector
            )
        adjoint_solution = derivative_solver.solve(adjoint_vector)
        gradient = (
            adjoint_solution.T @ sparse_partial_parameter @ self._vertex_to_simplex_matrix
        )
        return gradient

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        derivative_solver = self._cached_state.get_derivative_solver(parameter_vector)
        sparse_partial_parameter = self._cached_state.get_sparse_partial_parameter(parameter_vector)
        if derivative_solver is None or sparse_partial_parameter is None:
            derivative_solver, sparse_partial_parameter = self._set_up_derivative_structures(
                solution_vector, parameter_vector
            )
            self._cached_state.set_parameter_vector(parameter_vector)
            self._cached_state.set_derivative_solver(derivative_solver, parameter_vector)
            self._cached_state.set_sparse_partial_parameter(
                sparse_partial_parameter, parameter_vector
            )
        direction_vector_on_cells = self._vertex_to_simplex_matrix @ direction_vector
        rhs_vector = sparse_partial_parameter @ direction_vector_on_cells

        # TODO: The actual JVP solve should go into Eikonax
        system_matrix = derivative_solver.sparse_system_matrix.T
        permutation_matrix = derivative_solver.sparse_permutation_matrix
        permuted_right_hand_side = permutation_matrix @ rhs_vector
        permuted_solution = sps.linalg.spsolve_triangular(
            system_matrix, permuted_right_hand_side, lower=True, unit_diagonal=True
        )
        jacobian_vector_product = permutation_matrix.T @ permuted_solution
        return jacobian_vector_product

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    def _set_up_derivative_structures(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[eikonax_derivator.DerivativeSolver, sps.coo_matrix]:
        parameter_vector_on_cells = self._vertex_to_simplex_matrix @ parameter_vector
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector_on_cells)
        output_partial_solution, output_partial_tensor = (
            self._eikonax_derivatior.compute_partial_derivatives(
                solution_vector, tensor_field_instance
            )
        )
        tensor_partial_parameter = self._tensor_field.assemble_jacobian(parameter_vector_on_cells)
        output_partial_parameter = eikonax_linalg.contract_derivative_tensors(
            output_partial_tensor, tensor_partial_parameter
        )
        sparse_partial_solution = eikonax_linalg.convert_to_scipy_sparse(output_partial_solution)
        sparse_partial_parameter = eikonax_linalg.convert_to_scipy_sparse(output_partial_parameter)
        derivative_solver = eikonax_derivator.DerivativeSolver(
            solution_vector, sparse_partial_solution
        )
        return derivative_solver, sparse_partial_parameter
