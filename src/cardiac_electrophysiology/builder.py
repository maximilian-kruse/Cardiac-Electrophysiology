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
    noisy_data_path: Path | None = None
    fiber_ensemble_path: Path | None = None


@dataclass
class Strategies:
    ground_truth_strategy: str | None = "from_mesh"
    mean_strategy: str | None = "from_ground_truth"
    noisy_data_strategy: str | None = "from_ground_truth"


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
    noise_variance: np.ndarray[tuple[int], np.dtype[np.float64]] | float
    num_observations: int | None = None
    seed: int | None = None


@dataclass
class LoggerSettings:
    do_printing: bool
    write_mode: str


@dataclass
class PosteriorBuilderSettings:
    paths: Paths
    strategies: Strategies
    prior_parameters: PriorParameters
    eikonal_parameters: EikonalParameters
    observation_parameters: ObservationParameters
    logger_settings: LoggerSettings


@dataclass
class PosteriorBuilderOutput:
    pv_mesh: pv.UnstructuredGrid
    fiber_transformator: parameter.AngleFiberTransformator
    angle_transformator: parameter.AngleParameterTransformator
    ground_truth_parameter: np.ndarray[tuple[int], np.dtype[np.float64]]
    prior_mean_parameter: np.ndarray[tuple[int], np.dtype[np.float64]]
    observation_inds: np.ndarray[tuple[int], np.dtype[np.float64]]
    noisy_data: np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
class PosteriorBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: PosteriorBuilderSettings) -> None:
        self._paths = settings.paths
        self._strategies = settings.strategies
        self._prior_parameters = settings.prior_parameters
        self._eikonal_parameters = settings.eikonal_parameters
        self._observation_parameters = settings.observation_parameters
        self._logger_settings = settings.logger_settings

        self._pv_mesh = None
        self._dlx_mesh = None
        self._mesh_vertices = None
        self._mesh_simplices = None
        self._basis_vectors = None
        self._fiber_field_ensemble = None
        self._ground_truth_parameter = None
        self._mean_angle_field = None
        self._prior_mean_parameter = None
        self._observation_inds = None
        self._noisy_data = None
        self._fiber_transformator = None
        self._angle_transformator = None
        self._prior_component = None
        self._pts_map = None
        self._likelihood_component = None

    # ----------------------------------------------------------------------------------------------
    def build(self, return_additional_data: bool = False) -> None:
        self._pv_mesh, self._dlx_mesh, self._basis_vectors = self._load_mesh_data()
        self._mesh_vertices, self._mesh_simplices, self._ground_truth_fiber_field = (
            self._extract_mesh_data()
        )
        self._fiber_transformator = parameter.AngleFiberTransformator(
            self._basis_vectors[..., 0], self._basis_vectors[..., 1]
        )
        self._mean_angle_field = self._compute_mean_angle_field()
        self._angle_transformator = parameter.AngleParameterTransformator(self._mean_angle_field)
        self._prior_mean_parameter = self._angle_transformator.mean_parameter

        self._prior_component = self._create_prior()
        tensor_field_object = self._create_tensor_field()
        ekx_solver, ekx_derivator = self._create_eikonax_solver_and_derivator()
        self._pts_map = ptsmap.EikonalPTSMap(ekx_solver, ekx_derivator, tensor_field_object)
        self._observation_inds, self._noisy_data = self._get_observational_data()
        self._likelihood_component = self._create_likelihood()

        logger = self._create_logger()
        posterior_component = ls_bip_posterior.LogPosterior(
            self._likelihood_component, self._pts_map, self._prior_component, logger
        )
        if return_additional_data:
            additional_output = PosteriorBuilderOutput(
                pv_mesh = self._pv_mesh,
                fiber_transformator=self._fiber_transformator,
                angle_transformator=self._angle_transformator,
                ground_truth_parameter=self._ground_truth_parameter,
                prior_mean_parameter=self._prior_mean_parameter,
                observation_inds=self._observation_inds,
                noisy_data=self._noisy_data,
            )
            return posterior_component, additional_output
        else:
            return posterior_component

    # ----------------------------------------------------------------------------------------------
    def _load_mesh_data(
        self,
    ) -> tuple[
        pv.UnstructuredGrid, dlx.mesh.Mesh, np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ]:
        pv_mesh = pv.read(self._paths.vtu_mesh_path)
        with XDMFFile(MPI.COMM_WORLD, self._paths.xdmf_mesh_path, "r") as xdmf:
            dlx_mesh = xdmf.read_mesh(name="Grid")
        basis_vectors = np.load(self._paths.basis_vecs_path)

        return pv_mesh, dlx_mesh, basis_vectors

    # ----------------------------------------------------------------------------------------------
    def _extract_mesh_data(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]] | None,
    ]:
        vertices = self._pv_mesh.points
        simplices = self._pv_mesh.cells.reshape(-1, 4)[:, 1:]
        if self._strategies.ground_truth_strategy == "from_mesh":
            fiber_vectors = self._pv_mesh.cell_data["fibers"]
        else:
            fiber_vectors = None
        return vertices, simplices, fiber_vectors

    # ----------------------------------------------------------------------------------------------
    def _compute_mean_angle_field(self) -> np.ndarray:
        match self._strategies.mean_strategy:
            case "from_ground_truth":
                ground_truth_angle = self._fiber_transformator.compute_angle_from_fiber(
                    self._ground_truth_fiber_field
                )
                mean_angle = np.mean(ground_truth_angle) * np.ones(self._mesh_simplices.shape[0])
            case "from_ensemble":
                accumulated_field = sum(
                    self._fiber_field_ensemble[key] for key in self._fiber_field_ensemble.files
                )
                mean_fiber_field = accumulated_field / len(self._fiber_field_ensemble.files)
                mean_angle = self._fiber_transformator.compute_angle_from_fiber(mean_fiber_field)
        return mean_angle

    # ----------------------------------------------------------------------------------------------
    def _create_prior(self) -> prior.FiberFieldPrior:
        prior_settings = ls_prior_builder.BilaplacianPriorSettings(
            mesh=self._dlx_mesh,
            mean_vector=self._prior_mean_parameter,
            kappa=self._prior_parameters.kappa,
            tau=self._prior_parameters.tau,
            seed=self._prior_parameters.seed,
        )
        prior_component = prior.FiberFieldPrior(prior_settings)
        return prior_component

    # ----------------------------------------------------------------------------------------------
    def _create_tensor_field(self) -> eikonax_tensorfield.TensorField:
        longitudinal_velocity_vector = self._eikonal_parameters.longitudinal_velocity * np.ones(
            self._mesh_simplices.shape[0]
        )
        transversal_velocity_vector = self._eikonal_parameters.transversal_velocity * np.ones(
            self._mesh_simplices.shape[0]
        )
        fiber_tensor_settings = fibertensor.FiberTensorSettings(
            dimension=3,
            mean_angle_vector=self._mean_angle_field,
            basis_vectors_one=self._basis_vectors[..., 0],
            basis_vectors_two=self._basis_vectors[..., 1],
            longitudinal_velocities=longitudinal_velocity_vector,
            transversal_velocities=transversal_velocity_vector,
        )
        fiber_tensor = fibertensor.FiberTensor(fiber_tensor_settings)
        tensor_field_mapping = eikonax_tensorfield.LinearScalarMap()
        tensor_field_object = eikonax_tensorfield.TensorField(
            num_simplices=self._mesh_simplices.shape[0],
            vector_to_simplices_map=tensor_field_mapping,
            simplex_tensor=fiber_tensor,
        )
        return tensor_field_object

    # ----------------------------------------------------------------------------------------------
    def _create_eikonax_solver_and_derivator(
        self,
    ) -> tuple[eikonax_solver.Solver, eikonax_derivator.PartialDerivator]:
        ekx_solver_settings = eikonax_solver.SolverData(
            tolerance=self._eikonal_parameters.solver_tolerance,
            max_num_iterations=self._eikonal_parameters.max_num_iterations,
            max_value=self._eikonal_parameters.max_value,
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

        initial_sites = eikonax_preprocessing.InitialSites(
            inds=(self._eikonal_parameters.initial_site_ind,), values=(0,)
        )
        mesh_data = eikonax_preprocessing.MeshData(self._mesh_vertices, self._mesh_simplices)
        ekx_solver = eikonax_solver.Solver(mesh_data, ekx_solver_settings, initial_sites)
        ekx_derivator = eikonax_derivator.PartialDerivator(
            mesh_data, eikonax_derivator_settings, initial_sites
        )
        return ekx_solver, ekx_derivator

    # ----------------------------------------------------------------------------------------------
    def _get_observational_data(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        match self._strategies.noisy_data_strategy:
            case "from_file":
                observations = np.load(self._paths.noisy_data_path)
                observation_inds = observations["inds"]
                noisy_data = observations["data"]
            case "from_ground_truth":
                rng = np.random.default_rng(seed=self._observation_parameters.seed)
                observation_inds = rng.integers(
                    low=0,
                    high=self._mesh_vertices.shape[0],
                    size=self._observation_parameters.num_observations,
                )
                ground_truth_angle = self._fiber_transformator.compute_angle_from_fiber(
                    self._ground_truth_fiber_field
                )
                self._ground_truth_parameter = (
                    self._angle_transformator.compute_parameter_from_angle(ground_truth_angle)
                )
                ground_truth_solution = self._pts_map.evaluate_forward(self._ground_truth_parameter)
                noise = rng.normal(
                    loc=0.0,
                    scale=np.sqrt(self._observation_parameters.noise_variance),
                    size=self._observation_parameters.num_observations,
                )
                noisy_data = ground_truth_solution[observation_inds] + noise
        return observation_inds, noisy_data

    # ----------------------------------------------------------------------------------------------
    def _create_likelihood(self) -> ls_bip_components.Likelihood:
        precision_values = np.full_like(
            self._noisy_data, 1 / self._observation_parameters.noise_variance
        )
        observation_matrix = ls_bip_utilities.assemble_vertex_observation_matrix(
            self._mesh_vertices.shape[0], self._observation_inds
        )
        noise_precision_matrix = ls_bip_utilities.assemble_diagonal_precision_matrix(
            precision_values
        )
        log_likelihood = ls_bip_components.GaussianLogLikelihood(
            self._noisy_data, observation_matrix, noise_precision_matrix
        )
        return log_likelihood

    # ----------------------------------------------------------------------------------------------
    def _create_logger(self) -> ls_bip_logging.LSBIPLogger:
        logger_settings = ls_bip_logging.LoggerSettings(
            do_printing=self._logger_settings.do_printing,
            logfile_path=self._paths.log_file_path,
            write_mode=self._logger_settings.write_mode,
        )
        logger = ls_bip_logging.LSBIPLogger(logger_settings)
        return logger
