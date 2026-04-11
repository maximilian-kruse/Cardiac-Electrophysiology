from typing import override

import numpy as np
from eikonax import derivator as eikonax_derivator
from eikonax import linalg as eikonax_linalg
from eikonax import solver as eikonax_solver
from eikonax import tensorfield as eikonax_tensorfield

from cardiac_electrophysiology.ls_bip import components as ls_bip_components


# ==================================================================================================
class EikonalPTSMap(ls_bip_components.ParameterToSolutionMap):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        eikonax_solver: eikonax_solver.Solver,
        eikonax_derivatior: eikonax_derivator.PartialDerivator,
        tensor_field: eikonax_tensorfield.TensorField,
    ) -> None:
        self._eikonax_solver = eikonax_solver
        self._eikonax_derivatior = eikonax_derivatior
        self._tensor_field = tensor_field

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_forward(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
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
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector)
        output_partial_solution, output_partial_tensor = (
            self._eikonax_derivatior.compute_partial_derivatives(
                solution_vector, tensor_field_instance
            )
        )
        tensor_partial_parameter = self._tensor_field.assemble_jacobian(parameter_vector)
        output_partial_parameter = eikonax_linalg.contract_derivative_tensors(
            output_partial_tensor, tensor_partial_parameter
        )
        sparse_partial_solution = eikonax_linalg.convert_to_scipy_sparse(output_partial_solution)
        sparse_partial_parameter = eikonax_linalg.convert_to_scipy_sparse(output_partial_parameter)
        derivative_solver = eikonax_derivator.DerivativeSolver(
            solution_vector, sparse_partial_solution
        )
        adjoint_solution = derivative_solver.solve(adjoint_vector)
        gradient = adjoint_solution.T @ sparse_partial_parameter
        return gradient

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
