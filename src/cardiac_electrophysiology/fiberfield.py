from typing import final

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from eikonax import tensorfield
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
@final
class FiberTensor(tensorfield.AbstractSimplexTensor):
    _mean_parameter_vector: jtReal[jax.Array, "num_parameters"]
    _mean_angle_vector: jtReal[jax.Array, "num_parameters"]
    _basis_vectors: tuple[
        jtReal[jax.Array, "num_parameters dim"],
        jtReal[jax.Array, "num_parameters dim"],
    ]
    _conduction_velocities: tuple[
        jtReal[jax.Array, "num_parameters"],
        jtReal[jax.Array, "num_parameters"],
    ]

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        dimension: int,
        mean_parameter_vector: jtReal[jax.Array | npt.NDArray, "num_parameters"],
        basis_vectors: tuple[
            jtReal[jax.Array | npt.NDArray, "num_parameters dim"],
            jtReal[jax.Array | npt.NDArray, "num_parameters dim"],
        ],
        conduction_velocities: tuple[
            jtReal[jax.Array | npt.NDArray, "num_parameters"],
            jtReal[jax.Array | npt.NDArray, "num_parameters"],
        ],
    ) -> None:
        self.dimension = dimension
        self._mean_parameter_vector = mean_parameter_vector
        self._mean_angle_vector = jnp.arccos(jnp.tanh(mean_parameter_vector))
        self._basis_vectors = basis_vectors
        self._conduction_velocities = conduction_velocities

    # ----------------------------------------------------------------------------------------------
    def assemble(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameter: jtReal[jax.Array, "num_parameters_local"],
    ) -> jtReal[jax.Array, "dim dim"]:
        e_1 = self._basis_vectors[0][simplex_ind]
        e_2 = self._basis_vectors[1][simplex_ind]
        long_velocity = self._conduction_velocities[0][simplex_ind]
        trans_velocity = self._conduction_velocities[1][simplex_ind]
        mean_angle = self._mean_angle_vector[simplex_ind]
        angle = jnp.arccos(jnp.tanh(parameter)) + mean_angle - jnp.pi / 2
        long_vector = jnp.cos(angle) * e_1 + jnp.sin(angle) * e_2
        trans_vector = -jnp.sin(angle) * e_1 + jnp.cos(angle) * e_2
        tensor = 1 / jnp.square(long_velocity) * jnp.outer(
            long_vector, long_vector
        ) + 1 / jnp.square(trans_velocity) * jnp.outer(trans_vector, trans_vector)
        return tensor

    # ----------------------------------------------------------------------------------------------
    def derivative(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameter: jtReal[jax.Array, "num_parameters_local"],
    ) -> jtReal[jax.Array, "dim dim num_local_parameters"]:
        derivative_function = jax.jacfwd(self.assemble, argnums=1)
        tensor = derivative_function(simplex_ind, parameter)
        return tensor


# ==================================================================================================
def assemble_vertex_to_simplex_interpolation_matrix(mesh: pv.PolyData) -> sp.coo_array:
    num_vertices = mesh.number_of_points
    num_simplices = mesh.number_of_cells

    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:])
    row_inds = np.repeat(np.arange(num_simplices), 3)
    col_inds = simplices.flatten()
    data = np.full(num_simplices * 3, 1 / 3)
    interpolation_matrix = sp.coo_array(
        (data, (row_inds, col_inds)), shape=(num_simplices, num_vertices)
    )
    return interpolation_matrix
