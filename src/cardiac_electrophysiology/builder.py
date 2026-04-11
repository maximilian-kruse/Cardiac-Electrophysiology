from dataclasses import dataclass
from pathlib import Path

import dolfinx as dlx
import numpy as np
import pyvista as pv
from dolfinx.io import XDMFFile
from eikonax import derivator as eikonax_derivator
from eikonax import preprocessing as eikonax_preprocessing
from eikonax import solver as eikonax_solver
from eikonax import tensorfield as eikonax_tensorfield
from ls_prior import builder as ls_prior_builder
from mpi4py import MPI

from cardiac_electrophysiology.components import fibertensor, parameter, prior, ptsmap
from cardiac_electrophysiology.ls_bip import components as ls_bip_components
from cardiac_electrophysiology.ls_bip import logging as ls_bip_logging
from cardiac_electrophysiology.ls_bip import posterior as ls_bip_posterior
from cardiac_electrophysiology.ls_bip import utilities as ls_bip_utilities


# ==================================================================================================
@dataclass
class Paths:
    vtu_mesh_path: Path
    xdmf_mesh_path: Path
    basis_vecs_path: Path
    log_file_path: Path


@dataclass
class PriorParameters:
    kappa: float
    tau: float
    seed: int


@dataclass
class EikonalParameters:
    solver_tolerance: float
    max_num_iterations: int
    max_value: float
    initial_site_ind: int
    longitudinal_velocity: float
    transversal_velocity: float


@dataclass
class ObservationParameters:
    noise_variance: float
    num_observations: int


@dataclass
class LoggerSettings:
    do_printing: bool
    write_mode: str


@dataclass
class PosteriorBuilderSettings:
    paths: Paths
    prior_parameters: PriorParameters
    eikonal_parameters: EikonalParameters
    observation_parameters: ObservationParameters
    logger_settings: LoggerSettings


# ==================================================================================================
class PosteriorBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: PosteriorBuilderSettings) -> None:
        self._settings = settings

    # ----------------------------------------------------------------------------------------------
    def build(self) -> None:
        vertices, simplices, basis_vectors, fiber_vectors, dlx_mesh = self._load_data()
        fiber_transformator, angle_transformator = self._create_transformators(
            basis_vectors, fiber_vectors
        )
        prior_component, prior_mean_parameter = self._create_prior(
            dlx_mesh, angle_transformator, simplices.shape[0]
        )
        eikonal_pts_map = self._create_eikonal_ptsmap(
            vertices, simplices, angle_transformator.mean_angle, basis_vectors
        )
        likelihood_component, ground_truth_parameter = self._create_likelihood(
            fiber_transformator,
            angle_transformator,
            fiber_vectors,
            eikonal_pts_map,
            vertices.shape[0],
        )
        logger = self._create_logger()
        posterior_component = ls_bip_posterior.LogPosterior(
            likelihood_component, eikonal_pts_map, prior_component, logger
        )
        return (
            posterior_component,
            fiber_transformator,
            angle_transformator,
            ground_truth_parameter,
            prior_mean_parameter,
        )

    # ----------------------------------------------------------------------------------------------
    def _load_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dlx.mesh.Mesh]:
        basis_vectors = np.load(self._settings.paths.basis_vecs_path)
        pv_mesh = pv.read(self._settings.paths.vtu_mesh_path)
        with XDMFFile(MPI.COMM_WORLD, self._settings.paths.xdmf_mesh_path, "r") as xdmf:
            dlx_mesh = xdmf.read_mesh(name="Grid")
        vertices = pv_mesh.points
        simplices = pv_mesh.cells.reshape(-1, 4)[:, 1:4]
        fiber_vectors = np.array(pv_mesh.cell_data["fibers"])
        return vertices, simplices, basis_vectors, fiber_vectors, dlx_mesh

    # ----------------------------------------------------------------------------------------------
    def _create_transformators(
        self, basis_vectors: np.ndarray, fiber_vectors: np.ndarray
    ) -> tuple[parameter.AngleFiberTransformator, parameter.AngleParameterTransformator]:
        fiber_transformator = parameter.AngleFiberTransformator(
            basis_vectors[..., 0], basis_vectors[..., 1]
        )
        fiber_angles = fiber_transformator.compute_angle_from_fiber(fiber_vectors)
        mean_fiber_angle = np.mean(fiber_angles)
        angle_transformator = parameter.AngleParameterTransformator(mean_fiber_angle)
        return fiber_transformator, angle_transformator

    # ----------------------------------------------------------------------------------------------
    def _create_prior(
        self,
        dlx_mesh: dlx.mesh.Mesh,
        angle_transformator: parameter.AngleParameterTransformator,
        num_simplices: int,
    ) -> prior.FiberFieldPrior:
        mean_parameter_value = angle_transformator.compute_parameter_from_angle(
            angle_transformator.mean_angle
        )
        mean_parameter_vector = np.full(num_simplices, mean_parameter_value)
        prior_settings = ls_prior_builder.BilaplacianPriorSettings(
            mesh=dlx_mesh,
            mean_vector=mean_parameter_vector,
            kappa=1,
            tau=1,
            seed=0,
        )
        prior_component = prior.FiberFieldPrior(prior_settings)
        return prior_component, mean_parameter_vector

    # ----------------------------------------------------------------------------------------------
    def _create_eikonal_ptsmap(
        self,
        vertices: np.ndarray,
        simplices: np.ndarray,
        mean_fiber_angle: float,
        basis_vectors: np.ndarray,
    ) -> ptsmap.EikonalPTSMap:
        num_simplices = simplices.shape[0]
        mean_angle_vector = mean_fiber_angle * np.ones(num_simplices)
        longitudinal_velocity_vector = np.full_like(
            mean_angle_vector, self._settings.eikonal_parameters.longitudinal_velocity
        )
        transversal_velocity_vector = np.full_like(
            mean_angle_vector, self._settings.eikonal_parameters.transversal_velocity
        )
        fiber_tensor_settings = fibertensor.FiberTensorSettings(
            dimension=3,
            mean_angle_vector=mean_angle_vector,
            basis_vectors_one=basis_vectors[..., 0],
            basis_vectors_two=basis_vectors[..., 1],
            longitudinal_velocities=longitudinal_velocity_vector,
            transversal_velocities=transversal_velocity_vector,
        )
        ekx_solver_settings = eikonax_solver.SolverData(
            tolerance=self._settings.eikonal_parameters.solver_tolerance,
            max_num_iterations=self._settings.eikonal_parameters.max_num_iterations,
            max_value=self._settings.eikonal_parameters.max_value,
            loop_type="jitted_while",
            use_soft_update=True,
            softminmax_order=20,
            softminmax_cutoff=0.01,
        )
        eikonax_derivator_settings = eikonax_derivator.PartialDerivatorData(
            use_soft_update=True,
            softminmax_order=20,
            softminmax_cutoff=0.01,
        )
        fiber_tensor = fibertensor.FiberTensor(fiber_tensor_settings)
        tensor_field_mapping = eikonax_tensorfield.LinearScalarMap()
        tensor_field_object = eikonax_tensorfield.TensorField(
            num_simplices=num_simplices,
            vector_to_simplices_map=tensor_field_mapping,
            simplex_tensor=fiber_tensor,
        )
        initial_sites = eikonax_preprocessing.InitialSites(
            inds=(self._settings.eikonal_parameters.initial_site_ind,), values=(0,)
        )
        mesh_data = eikonax_preprocessing.MeshData(vertices, simplices)
        ekx_solver = eikonax_solver.Solver(mesh_data, ekx_solver_settings, initial_sites)
        ekx_derivator = eikonax_derivator.PartialDerivator(
            mesh_data, eikonax_derivator_settings, initial_sites
        )
        eikonal_pts_map = ptsmap.EikonalPTSMap(
            ekx_solver, ekx_derivator, tensor_field_object
        )
        return eikonal_pts_map

    # ----------------------------------------------------------------------------------------------
    def _create_likelihood(
        self,
        fiber_transformator: parameter.AngleFiberTransformator,
        angle_transformator: parameter.AngleParameterTransformator,
        fiber_vectors: np.ndarray,
        eikonal_pts_map: ptsmap.EikonalPTSMap,
        num_vertices: int,
    ) -> ls_bip_components.Likelihood:
        fiber_angles = fiber_transformator.compute_angle_from_fiber(fiber_vectors)
        ground_truth_parameter = angle_transformator.compute_parameter_from_angle(fiber_angles)
        ground_truth_solution = eikonal_pts_map.evaluate_forward(ground_truth_parameter)
        rng = np.random.default_rng(seed=0)
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(self._settings.observation_parameters.noise_variance),
            size=num_vertices,
        )
        noisy_solution = ground_truth_solution + noise
        observation_inds = rng.integers(
            low=0,
            high=num_vertices,
            size=self._settings.observation_parameters.num_observations,
        )
        observations = noisy_solution[observation_inds]
        precision_values = np.full(
            self._settings.observation_parameters.num_observations,
            1 / self._settings.observation_parameters.noise_variance,
        )
        observation_matrix = ls_bip_utilities.assemble_vertex_observation_matrix(
            num_vertices, observation_inds
        )
        noise_precision_matrix = ls_bip_utilities.assemble_diagonal_precision_matrix(
            precision_values
        )
        log_likelihood = ls_bip_components.GaussianLogLikelihood(
            observations, observation_matrix, noise_precision_matrix
        )
        return log_likelihood, ground_truth_parameter

    # ----------------------------------------------------------------------------------------------
    def _create_logger(self) -> ls_bip_logging.LSBIPLogger:
        logger_settings = ls_bip_logging.LoggerSettings(
            do_printing=self._settings.logger_settings.do_printing,
            logfile_path=self._settings.paths.log_file_path,
            write_mode=self._settings.logger_settings.write_mode,
        )
        logger = ls_bip_logging.LSBIPLogger(logger_settings)
        return logger
