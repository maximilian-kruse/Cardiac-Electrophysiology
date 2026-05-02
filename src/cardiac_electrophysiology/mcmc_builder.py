from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_mcmc import algorithms, logging, model, output, sampling, storage

from .ls_bip import laplace, posterior


# ==================================================================================================
class CardiacEPMCMCModel(model.MCMCModel):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        log_posterior: posterior.LogPosterior,
        laplace_approximation: laplace.LaplaceApproximation,
        reference_point: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        self._reference_point = reference_point
        self._laplace_approximation = laplace_approximation
        self._posterior = log_posterior

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        likelihood_potential, _ = self._posterior.evaluate_cost(state, split=True)
        return likelihood_potential

    # ----------------------------------------------------------------------------------------------
    @override
    def compute_preconditioner_sqrt_action(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = self._posterior.prior.apply_covariance_factorization(random_vector)
        return result

    # ----------------------------------------------------------------------------------------------
    def compute_preconditioner_inv_action(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = self._laplace_approximation.apply_precision(vector)
        return result

    # ----------------------------------------------------------------------------------------------
    @override
    @property
    def reference_point(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._reference_point

    # ----------------------------------------------------------------------------------------------
    @override
    @property
    def random_vector_size(self) -> int:
        return self._posterior.prior.random_vector_size


# ==================================================================================================
@dataclass
class MCMCModelSettings:
    log_posterior: posterior.LogPosterior
    laplace_approximation: laplace.LaplaceApproximation
    reference_point: np.ndarray[tuple[int], np.dtype[np.float64]]
    step_width: float
    index_to_track: int


@dataclass
class MCMCBuilderSettings:
    mcmc_model_settings: MCMCModelSettings
    logging_settings: logging.LoggerSettings
    storage_path: Path
    storage_chunk_size: int
    overwrite_existing_storage: bool


# ==================================================================================================
class MCMCBuilder:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: MCMCBuilderSettings) -> None:
        self._mcmc_model_settings = settings.mcmc_model_settings
        self._logging_settings = settings.logging_settings
        self._storage_path = settings.storage_path
        self._storage_chunk_size = settings.storage_chunk_size
        self._overwrite_existing_storage = settings.overwrite_existing_storage

    # ----------------------------------------------------------------------------------------------
    def build(self) -> sampling.Sampler:
        mcmc_model = CardiacEPMCMCModel(
            log_posterior=self._mcmc_model_settings.log_posterior,
            reference_point=self._mcmc_model_settings.reference_point,
            laplace_approximation=self._mcmc_model_settings.laplace_approximation,
        )
        algorithm = algorithms.pCNAlgorithm(mcmc_model, self._mcmc_model_settings.step_width)
        sample_storage = storage.ZarrStorage(
            save_directory=self._storage_path,
            chunk_size=self._storage_chunk_size,
            overwrite=self._overwrite_existing_storage,
        )
        logger = logging.MCMCLogger(self._logging_settings)
        outputs = self._create_outputs()
        sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)
        return sampler

    # ----------------------------------------------------------------------------------------------
    def _create_outputs(self) -> list[output.MCMCOutput]:
        acceptance_rate_output = output.MCMCOutput(
            output.AcceptanceQoI(),
            output.RunningMeanStatistic(),
            f"{'Accept Rate':<15}",
            "<+15.3e",
            log=True,
        )
        component_output = output.MCMCOutput(
            output.ComponentQoI(self._mcmc_model_settings.index_to_track),
            output.IdentityStatistic(),
            f"{f'Component {self._mcmc_model_settings.index_to_track}':<15}",
            "<+15.3e",
            log=True,
        )
        running_mean_component_output = output.MCMCOutput(
            output.ComponentQoI(self._mcmc_model_settings.index_to_track),
            output.RunningMeanStatistic(),
            f"{f'Run_mean_C_{self._mcmc_model_settings.index_to_track}':<15}",
            "<+15.3e",
            log=True,
        )
        outputs = (acceptance_rate_output, component_output, running_mean_component_output)
        return outputs
