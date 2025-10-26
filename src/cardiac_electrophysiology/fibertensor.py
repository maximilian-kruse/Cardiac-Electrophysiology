from typing import final

import jax
import jax.numpy as jnp
import numpy.typing as npt
from eikonax import tensorfield
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
@final
class FiberTensor(tensorfield.AbstractSimplexTensor):
    _mean_parameter_vector: jtReal[jax.Array, "num_parameters"]
    _mean_angle_vector: jtReal[jax.Array, "num_parameters"]
    _first_basis_vectors: jtReal[jax.Array, "num_parameters dim"]
    _second_basis_vectors: jtReal[jax.Array, "num_parameters dim"]
    _longitudinal_velocities: jtReal[jax.Array, "num_parameters"]
    _transversal_velocities: jtReal[jax.Array, "num_parameters"]

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        dimension: int,
        mean_angle_vector: jtReal[jax.Array | npt.NDArray, "num_parameters"],
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
        self._mean_angle_vector = jnp.array(mean_angle_vector, dtype=jnp.float32)
        self._mean_parameter_vector = jnp.arctanh(jnp.cos(self._mean_angle_vector + jnp.pi / 2))
        self._first_basis_vectors = jnp.array(basis_vectors[0], dtype=jnp.float32)
        self._second_basis_vectors = jnp.array(basis_vectors[1], dtype=jnp.float32)
        self._longitudinal_velocities = jnp.array(conduction_velocities[0], dtype=jnp.float32)
        self._transversal_velocities = jnp.array(conduction_velocities[1], dtype=jnp.float32)

    # ----------------------------------------------------------------------------------------------
    def assemble(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameter: jtReal[jax.Array, "num_parameters_local"],
    ) -> jtReal[jax.Array, "dim dim"]:
        e_1 = self._first_basis_vectors[simplex_ind]
        e_2 = self._second_basis_vectors[simplex_ind]
        long_velocity = self._longitudinal_velocities[simplex_ind]
        trans_velocity = self._transversal_velocities[simplex_ind]
        mean_angle = self._mean_angle_vector[simplex_ind]
        centered_parameter = parameter - self._mean_parameter_vector[simplex_ind]
        angle = jnp.arccos(jnp.tanh(centered_parameter)) + mean_angle - jnp.pi / 2
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
