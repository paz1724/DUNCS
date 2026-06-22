from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import sys

import numpy as np
import torch

from src.config.simulation_config import SimulationConfig
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.signal_creation import Samples
from src.training import TrainingParams, train
from src.evaluation import evaluate
from src.data_handler import create_dataset, load_datasets, SameLengthBatchSampler, collate_fn
from src.training import set_criterions
from src.utils import print_loss_results_from_simulation
from src.steering_vector_generator import SteeringVectorGenerator


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        """Initialize the simulation runner and prepare output directories.

        Args:
            config (SimulationConfig): Parsed simulation configuration.
        """
        self.config = config
        self.base_path = Path(__file__).parent / "data"
        self.paths = self._init_paths()
        self.monte_carlo_simulations = 1


    def _init_paths(self):
        """Build and create the dataset/weights/simulations directory layout.

        Returns:
            dict: Mapping of path names ('datasets', 'saving', 'simulations') to Path objects.
        """
        paths = {
            "datasets": self.base_path / "datasets" / "uniform_bias_spacing",
            "saving": self.base_path / "weights",
            "simulations": self.base_path / "simulations"
        }
        for p in [paths["datasets"] / "train", paths["datasets"] / "test", paths["saving"] / "final_models"]:
            p.mkdir(parents=True, exist_ok=True)
        return paths

    def train_model(self, model_gen, train_dataset):
        """Build training parameters, train the model, and optionally save weights.

        Args:
            model_gen (ModelGenerator): Factory holding the instantiated model.
            train_dataset (Dataset): Materialized training dataset.

        Returns:
            nn.Module: The trained model (best weights loaded).
        """
        config = self.config
        simulation_parameters = (
            TrainingParams()
            .set_training_objective(config.training.training_objective)
            .set_batch_size(config.training.batch_size)
            .set_epochs(config.training.epochs)
            .set_model(model_gen=model_gen)
            .set_optimizer(config.training.optimizer,
                           config.training.learning_rate,
                           config.training.weight_decay)
            .set_training_dataset(train_dataset)
            .set_schedular(config.training.scheduler,
                           config.training.step_size,
                           config.training.gamma,
                           np.ceil(((0.9*len(train_dataset)) / config.training.batch_size))*config.training.epochs)
        )

        model, _train_res = train(
            training_parameters=simulation_parameters,
            saving_path=self.paths["saving"],
            save_figures=config.commands.save_plots,
            plot_curves=config.commands.plot_results
        )

        if config.commands.save_model:
            torch.save(model.state_dict(),
                       self.paths["saving"] / "final_models" / model.get_model_file_name())

        return model

    def evaluate_model(self, model, system_model, test_dataset):
        """Evaluate a trained model plus configured baselines on the test set.

        Args:
            model (nn.Module): Trained model to evaluate (may be None).
            system_model (SystemModel): Array geometry / steering model.
            test_dataset (Dataset): Materialized test dataset.

        Returns:
            dict: Nested results dict keyed by criterion and method name.
        """
        config = self.config
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_sampler=SameLengthBatchSampler(test_dataset, batch_size=128),
            shuffle=False,
            pin_memory=True,
        )

        criterions = set_criterions(
            config.evaluation.criterion,
            system_model.array,
        )

        return evaluate(
            generic_test_dataset=test_loader,
            criterions=criterions,
            system_model=system_model,
            models=config.evaluation.models,
            augmented_methods=config.evaluation.augmented_methods,
            subspace_methods=config.evaluation.subspace_methods,
            model_tmp=model,
            cov_recon_method=config.evaluation.covariance_reconstruction,
            cov_recon_params=asdict(config.admm_params),
            admm_iterations=config.evaluation.admm_iterations
        )

    def get_dataset(self, create_data, samples_model):
        """Create or load the train and/or test datasets per the active commands.

        Args:
            create_data (bool): If True, generate datasets; otherwise load from disk.
            samples_model (Samples): Signal sample generator for the system model.

        Returns:
            tuple: (train_dataset, test_dataset), either of which may be None
                depending on the train_model / evaluate_mode commands.
        """
        config = self.config
        train_dataset = None
        test_dataset = None

        if create_data:
            print("creating dataset...")
            if config.commands.train_model:
                train_dataset = create_dataset(
                    samples_model, config.training.samples_size, config.commands.save_dataset,
                    self.paths["datasets"], config.training.true_doa_train,
                    config.training.true_range_train, phase="train"
                )

            if config.commands.evaluate_mode:
                test_dataset = create_dataset(
                    samples_model, int(config.training.train_test_ratio * config.training.samples_size),
                    config.commands.save_dataset,
                    self.paths["datasets"], config.training.true_doa_test,
                    config.training.true_range_test, phase="test"
                )
        else:
            print("Loading dataset...")
            if config.commands.train_model:
                train_dataset = load_datasets(
                    config.system_model,
                    config.training.samples_size,
                    self.paths["datasets"],
                    is_training=True
                )

            if config.commands.evaluate_mode:
                test_dataset = load_datasets(
                    config.system_model,
                    int(config.training.train_test_ratio * config.training.samples_size),
                    self.paths["datasets"],
                    is_training=False
                )
        return train_dataset, test_dataset

    def _run_single_simulation(self, create_dataset):
        """Run one full create-data -> train -> evaluate pass for the current config.

        Args:
            create_dataset (bool): Whether to generate fresh datasets for this run.

        Returns:
            dict or None: Evaluation results dict if evaluate_mode is enabled, else None.
        """
        config = self.config

        # Redirect stdout to file if enabled
        if config.commands.save_to_file:
            now = datetime.now().strftime("%d_%m_%Y_%H_%M")
            suffix = ""
            if config.commands.train_model:
                suffix += f"_train_{config.model.model_type}_{config.training.training_objective}"
            suffix += (f"_{config.system_model.signal_nature}_SNR_{config.system_model.snr}_T_{config.system_model.T}"\
                       f"_eta{config.system_model.eta}.txt")
            log_file = self.paths["simulations"] / "results" / "scores" / Path(now + suffix)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.orig_stdout = sys.stdout
            sys.stdout = open(log_file, "w")

        system_model = SystemModel(config.system_model)
        model_gen = (
            ModelGenerator()
            .set_model_type(config.model.model_type)
            .set_system_model(system_model)
            .set_model_params(config.model.model_params)
            .set_model()
        )
        samples_model = Samples(config.system_model, config.system_model.antenna_pattern)
        train_dataset, test_dataset = self.get_dataset(create_dataset, samples_model)

        model = None
        if config.commands.train_model:
            train_dataset.materialize(config.system_model)
            model = self.train_model(model_gen, train_dataset)

        result = None
        if config.commands.evaluate_mode:
            test_dataset.materialize(config.system_model)
            result = self.evaluate_model(model, system_model, test_dataset)

        if config.commands.save_to_file:
            sys.stdout.close()
            sys.stdout = self.orig_stdout
            
        SteeringVectorGenerator.reset_instance()

        return result

    def run(self):
        """Run the simulation, either once or swept over the configured scenario.

        Returns:
            dict or None: For a single run, the evaluation results dict (or None).
                For a scenario sweep, a nested dict keyed by scenario parameter and
                value holding the averaged losses.
        """
        create_data = self.config.commands.create_data
        if not self.config.scenario:
            return self._run_single_simulation(create_data)

        loss_dict = {}
        for key, values in self.config.scenario.items():
            if key == 'eta':
                create_data = True
            loss_dict[key] = {}
            for val in values:
                setattr(self.config.system_model, key, val)
                loss, successful_simulations = {}, 0
                for i in range(self.monte_carlo_simulations):
                    print(f"Running scenario: {key} = {val}, simulation step = {i}")
                    result = self._run_single_simulation(create_data)
                    if result is not None:
                        successful_simulations += 1
                        if not loss:
                            loss = result
                        else:
                            for k in result.keys():
                                for test in result[k].keys():
                                    for res_type in result[k][test].keys():
                                        loss[k][test][res_type] += result[k][test][res_type]
                    if create_data and self.config.commands.save_dataset and key != 'eta':
                        create_data = False
                for k in loss.keys():  # Iterate over loss functions
                    for test in loss[k].keys():
                        for res_type in loss[k][test].keys():
                            loss[k][test][res_type] = loss[k][test][res_type] / successful_simulations
                loss_dict[key][val] = loss
                print(loss_dict)

        if None not in list(next(iter(loss_dict.values())).values()):
            print_loss_results_from_simulation(loss_dict)
            # if self.config.commands.plot_results:
            #     plot_results(
            #         loss_dict,
            #         criterion=self.config.evaluation.criterion,
            #         plot_acc=False,
            #         save_to_file=self.config.commands.save_plots,
            #     )

        return loss_dict
