"""
Compare DUNCS vs SparseNet: train both models on the same dataset,
produce comparison plots of training/validation loss and accuracy,
and evaluate both on the same test set.

Usage:
    python compare_models.py
"""

from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.signal_creation import Samples
from src.data_handler import create_dataset, SameLengthBatchSampler, collate_fn
from src.training import TrainingParams, train, plot_comparison_curves, plot_final_comparison_bar
from src.evaluation import evaluate_dnn_model, evaluate_model_based
from src.methods_pack.cov_reconstruct import get_cov_reconstruction_method
from src.metrics.criterions import set_criterions
from src.steering_vector_generator import SteeringVectorGenerator


def build_training_params(model_gen, train_dataset, config):
    """Build TrainingParams from a config and model generator."""
    return (
        TrainingParams()
        .set_training_objective(config.training.training_objective)
        .set_batch_size(config.training.batch_size)
        .set_epochs(config.training.epochs)
        .set_model(model_gen=model_gen)
        .set_optimizer(
            config.training.optimizer,
            config.training.learning_rate,
            config.training.weight_decay,
        )
        .set_training_dataset(train_dataset)
        .set_schedular(
            config.training.scheduler,
            config.training.step_size,
            config.training.gamma,
            np.ceil((0.9 * len(train_dataset)) / config.training.batch_size)
            * config.training.epochs,
        )
    )


def main():
    base_path = Path(__file__).parent / "data"
    weights_path = base_path / "weights"
    plots_path = base_path / "simulations" / "results" / "plots"
    weights_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load both configs ----
    duncs_config = load_simulation_config("src/config/DUNCS.yaml")
    sparse_config = load_simulation_config("src/config/sparseNet.yaml")

    # ---- 2. Shared dataset (same system model, max sample size) ----
    samples_size = max(duncs_config.training.samples_size,
                       sparse_config.training.samples_size)
    test_size = int(duncs_config.training.train_test_ratio * samples_size)

    print(f"=== Creating shared dataset: {samples_size} train, {test_size} test ===")
    system_model = SystemModel(duncs_config.system_model)
    samples_model = Samples(duncs_config.system_model, duncs_config.system_model.antenna_pattern)

    train_dataset = create_dataset(samples_model, samples_size, phase="train")
    test_dataset = create_dataset(samples_model, test_size, phase="test")

    # Materialize with SNR/T
    train_dataset.materialize(duncs_config.system_model)
    test_dataset.materialize(duncs_config.system_model)

    # Test loader (shared)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_sampler=SameLengthBatchSampler(test_dataset, batch_size=128),
        shuffle=False,
        pin_memory=True,
    )

    histories = {}
    models = {}

    # ---- 3. Train DUNCS ----
    print("\n" + "=" * 60)
    print("  TRAINING DUNCS")
    print("=" * 60)
    duncs_model_gen = (
        ModelGenerator()
        .set_model_type("DUNCS")
        .set_system_model(system_model)
        .set_model_params(duncs_config.model.model_params)
        .set_model()
    )
    duncs_params = build_training_params(duncs_model_gen, train_dataset, duncs_config)
    duncs_model, duncs_res = train(
        training_parameters=duncs_params,
        saving_path=weights_path,
        plot_curves=False,
        save_figures=False,
    )
    histories["DUNCS"] = duncs_res
    models["DUNCS"] = duncs_model

    # Reset steering vector cache between models
    SteeringVectorGenerator.reset_instance()

    # ---- 4. Train SparseNet ----
    print("\n" + "=" * 60)
    print("  TRAINING SparseNet")
    print("=" * 60)
    # SparseNet needs its own SystemModel (same params, fresh instance)
    system_model_sparse = SystemModel(sparse_config.system_model)
    sparse_model_gen = (
        ModelGenerator()
        .set_model_type("SparseNet")
        .set_system_model(system_model_sparse)
        .set_model_params(sparse_config.model.model_params)
        .set_model()
    )
    sparse_params = build_training_params(sparse_model_gen, train_dataset, sparse_config)
    sparse_model, sparse_res = train(
        training_parameters=sparse_params,
        saving_path=weights_path,
        plot_curves=False,
        save_figures=False,
    )
    histories["SparseNet"] = sparse_res
    models["SparseNet"] = sparse_model

    # ---- 5. Comparison plots ----
    print("\n" + "=" * 60)
    print("  GENERATING COMPARISON PLOTS")
    print("=" * 60)
    plot_comparison_curves(histories, save_dir=plots_path)

    # ---- 6. Evaluate both on test set ----
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)
    eval_results = {}

    for name, model in models.items():
        result = evaluate_dnn_model(model, test_loader, mode="test")
        eval_results[name] = result
        print(f"{name}: loss={result['loss']:.6f}, accuracy={result.get('Accuracy')}")

    # Also evaluate classical ESPRIT as baseline
    criterions = set_criterions(duncs_config.evaluation.criterion, system_model.array)
    cov_recon = get_cov_reconstruction_method("averaging", system_model)
    esprit_result = evaluate_model_based(
        test_loader, system_model, criterions[0], "esprit", cov_recon
    )
    eval_results["ESPRIT (classical)"] = esprit_result
    print(f"ESPRIT (classical): loss={esprit_result['loss']:.6f}")

    # ---- 7. Final bar chart ----
    plot_final_comparison_bar(eval_results, save_dir=plots_path)

    # ---- 8. Summary table ----
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'Test RMSPE':>12} {'Accuracy':>12}")
    print("-" * 50)
    for name, result in eval_results.items():
        acc_str = f"{result['Accuracy']:.2f}%" if result.get('Accuracy') is not None else "N/A"
        print(f"{name:<25} {result['loss']:>12.6f} {acc_str:>12}")

    print(f"\nPlots saved to: {plots_path}")
    print("Done!")


if __name__ == "__main__":
    main()
