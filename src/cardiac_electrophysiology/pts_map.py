from typing import override

import numpy as np
import scipy.sparse as sp
from eikonax import derivator, linalg, solver, tensorfield

from .ls_bip import components


# ==================================================================================================
class EikonalPTSMap(components.ParameterToSolutionMap):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        eikonax_solver: solver.Solver,
        eikonax_derivatior: derivator.PartialDerivator,
        tensor_field: tensorfield.TensorField,
        vertex_to_simplex_matrix: sp.coo_array,
        simplex_to_vertex_matrix: sp.coo_array,
    ) -> None:
        self._eikonax_solver = eikonax_solver
        self._eikonax_derivatior = eikonax_derivatior
        self._tensor_field = tensor_field
        self._vertex_to_simplex_matrix = vertex_to_simplex_matrix
        self._simplex_to_vertex_matrix = simplex_to_vertex_matrix

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_forward(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        parameter_vector = self._vertex_to_simplex_matrix @ parameter_vector
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector)
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
        parameter_vector = self._vertex_to_simplex_matrix @ parameter_vector
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector)
        output_partial_solution, output_partial_tensor = (
            self._eikonax_derivatior.compute_partial_derivatives(
                solution_vector, tensor_field_instance
            )
        )
        tensor_partial_parameter = self._tensor_field.assemble_jacobian(parameter_vector)
        output_partial_parameter = linalg.contract_derivative_tensors(
            output_partial_tensor, tensor_partial_parameter
        )
        sparse_partial_solution = linalg.convert_to_scipy_sparse(output_partial_solution)
        sparse_partial_parameter = linalg.convert_to_scipy_sparse(output_partial_parameter)
        derivative_solver = derivator.DerivativeSolver(solution_vector, sparse_partial_solution)
        adjoint_solution = derivative_solver.solve(adjoint_vector)
        gradient = adjoint_solution.T @ sparse_partial_parameter
        gradient = self._simplex_to_vertex_matrix @ gradient
        return None
