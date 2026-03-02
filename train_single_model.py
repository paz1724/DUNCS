"""
Train a single model and export raw training/evaluation data as JSON.

Called by the MATLAB wrapper (run_comparison.m) or standalone.

Usage:
    python train_single_model.py --model_name DUNCS --config_path src/config/DUNCS.yaml \
        --samples_size 20480 --test_ratio 0.1 --output_dir data/simulations/results

Output:
    <output_dir>/<model_name>_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.signal_creation import Samples
from src.data_handler import create_dataset, SameLengthBatchSampler, collate_fn
from src.training import TrainingParams, train
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
    parser = argparse.ArgumentParser(description="Train a single model and export results as JSON.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., DUNCS, SparseNet)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--samples_size", type=int, default=20480, help="Number of training samples")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test/train ratio")
    parser.add_argument("--output_dir", type=str, default="data/simulations/results", help="Output directory for JSON results")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to saved dataset (for sharing across models)")
    args = parser.parse_args()

    base_path = Path(__file__).parent / "data"
    weights_path = base_path / "weights"
    output_dir = Path(args.output_dir)
    dataset_cache_dir = base_path / "datasets" / "comparison_cache"
    weights_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load config ----
    config = load_simulation_config(args.config_path)
    test_size = int(args.test_ratio * args.samples_size)

    # ---- 2. Create or load shared dataset ----
    train_cache = dataset_cache_dir / f"train_{args.samples_size}.pt"
    test_cache = dataset_cache_dir / f"test_{test_size}.pt"

    system_model = SystemModel(config.system_model)
    samples_model = Samples(config.system_model, config.system_model.antenna_pattern)

    if train_cache.exists() and test_cache.exists():
        print(f"Loading cached dataset from {dataset_cache_dir}")
        train_dataset = torch.load(train_cache, weights_only=False)
        test_dataset = torch.load(test_cache, weights_only=False)
    else:
        print(f"Creating dataset: {args.samples_size} train, {test_size} test")
        train_dataset = create_dataset(samples_model, args.samples_size, phase="train")
        test_dataset = create_dataset(samples_model, test_size, phase="test")
        torch.save(train_dataset, train_cache)
        torch.save(test_dataset, test_cache)
        print(f"Dataset cached to {dataset_cache_dir}")

    train_dataset.materialize(config.system_model)
    test_dataset.materialize(config.system_model)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_sampler=SameLengthBatchSampler(test_dataset, batch_size=128),
        shuffle=False,
        pin_memory=True,
    )

    # ---- 3. Build and train model ----
    print(f"\n{'='*60}")
    print(f"  TRAINING {args.model_name}")
    print(f"{'='*60}")

    model_gen = (
        ModelGenerator()
        .set_model_type(config.model.model_type)
        .set_system_model(system_model)
        .set_model_params(config.model.model_params)
        .set_model()
    )
    training_params = build_training_params(model_gen, train_dataset, config)
    model, train_res = train(
        training_parameters=training_params,
        saving_path=weights_path,
        plot_curves=False,
        save_figures=False,
    )

    # ---- 4. Evaluate on test set ----
    print(f"\n{'='*60}")
    print(f"  EVALUATING {args.model_name}")
    print(f"{'='*60}")

    test_result = evaluate_dnn_model(model, test_loader, mode="test")
    print(f"{args.model_name}: test_rmspe={test_result['loss']:.6f}, accuracy={test_result.get('Accuracy')}")

    # Also evaluate classical ESPRIT as baseline
    criterions = set_criterions(config.evaluation.criterion, system_model.array)
    cov_recon = get_cov_reconstruction_method("averaging", system_model)
    esprit_result = evaluate_model_based(
        test_loader, system_model, criterions[0], "esprit", cov_recon
    )
    print(f"ESPRIT (classical): test_rmspe={esprit_result['loss']:.6f}")

    # ---- 5. Export results as JSON ----
    num_epochs = len(train_res["loss_train_list"])
    output = {
        "model_name": args.model_name,
        "config_path": args.config_path,
        "training": {
            "epochs": list(range(1, num_epochs + 1)),
            "loss_train": train_res["loss_train_list"],
            "loss_valid": train_res["loss_valid_list"],
            "acc_train": train_res.get("acc_train_list", []),
            "acc_valid": train_res.get("acc_valid_list", []),
        },
        "evaluation": {
            "test_rmspe": test_result["loss"],
            "test_accuracy": test_result.get("Accuracy"),
            "esprit_rmspe": esprit_result["loss"],
        },
        "system_model": {
            "N": config.system_model.N,
            "M": config.system_model.M,
            "T": config.system_model.T,
            "snr": config.system_model.snr,
            "signal_nature": config.system_model.signal_nature,
            "field_type": config.system_model.field_type,
        },
        "training_config": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "samples_size": args.samples_size,
        },
    }

    output_file = output_dir / f"{args.model_name}_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults exported to: {output_file}")

    SteeringVectorGenerator.reset_instance()
    return output


if __name__ == "__main__":
    main()
