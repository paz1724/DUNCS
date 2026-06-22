"""
Train a single model and export raw training/evaluation data as .mat file.

Called by the MATLAB wrapper (run_comparison.m) or standalone.

Usage:
    python train_single_model.py --model_name DUNCS --config_path src/config/DUNCS.yaml \
        --samples_size 20480 --test_ratio 0.1 --output_dir data/simulations/results

Output:
    <output_dir>/<model_name>_results.mat
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.signal_creation import Samples
from src.data_handler import create_dataset, partition_recorded_angles, SameLengthBatchSampler, collate_fn
from src.training import TrainingParams, train
from src.evaluation import evaluate_dnn_model, evaluate_model_based
from src.methods_pack.cov_reconstruct import get_cov_reconstruction_method
from src.metrics.criterions import set_criterions
from src.steering_vector_generator import SteeringVectorGenerator


def build_training_params(model_gen, train_dataset, config, valid_dataset=None):
    """Build TrainingParams from a config and model generator.

    If valid_dataset is given it is used as an AoA-disjoint validation set
    (no internal random split); otherwise the legacy 90/10 random split applies.

    Args:
        model_gen (ModelGenerator): Factory holding the instantiated model.
        train_dataset (Dataset): The training dataset.
        config (SimulationConfig): Parsed configuration providing training params.
        valid_dataset (Dataset, optional): Pre-built AoA-disjoint validation set.

    Returns:
        TrainingParams: Fully configured training parameters builder.
    """
    train_frac = 1.0 if valid_dataset is not None else 0.9
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
        .set_training_dataset(train_dataset, valid_dataset=valid_dataset)
        .set_schedular(
            config.training.scheduler,
            config.training.step_size,
            config.training.gamma,
            np.ceil((train_frac * len(train_dataset)) / config.training.batch_size)
            * config.training.epochs,
        )
    )


def main():
    """Parse CLI args, train one model, evaluate it, and export results to a .mat file.

    Reads command-line arguments (model name, config path, sample sizes, output dir),
    creates or loads the dataset, trains the model, evaluates it (plus a classical
    ESPRIT baseline) on the test set, and saves training curves and scalar metrics.

    Returns:
        dict: The result dictionary that was written to the .mat file.
    """
    parser = argparse.ArgumentParser(description="Train a single model and export results as .mat file.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., DUNCS, SparseNet)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--samples_size", type=int, default=20480, help="Number of training samples")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test/train ratio")
    parser.add_argument("--output_dir", type=str, default="data/simulations/results", help="Output directory for .mat results")
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

    # ---- 2. Create dataset(s) ----
    system_model = SystemModel(config.system_model)
    samples_model = Samples(config.system_model, config.system_model.antenna_pattern)

    use_aoa_disjoint = bool(getattr(config.system_model, "antenna_pattern", False)
                            and getattr(samples_model, "pattern_data", None))

    if use_aoa_disjoint:
        # AoA-disjoint train/val/test drawn from the recorded angle grid:
        # each split gets its own disjoint pool, so no angle is ever shared.
        val_size = test_size
        tr_a, va_a, te_a = partition_recorded_angles(
            samples_model, config.system_model.doa_range, ratios=(0.7, 0.15, 0.15))
        print(f"AoA-disjoint pools (recorded grid): "
              f"train {len(tr_a)} / val {len(va_a)} / test {len(te_a)} angles")
        train_dataset = create_dataset(samples_model, args.samples_size, phase="train", angle_pool=tr_a)
        valid_dataset = create_dataset(samples_model, val_size, phase="valid", angle_pool=va_a)
        test_dataset = create_dataset(samples_model, test_size, phase="test", angle_pool=te_a)
        for ds in (train_dataset, valid_dataset, test_dataset):
            ds.materialize(config.system_model)
    else:
        # Legacy: cached dataset + internal random 90/10 split (angle-agnostic)
        valid_dataset = None
        train_cache = dataset_cache_dir / f"train_{args.samples_size}.pt"
        test_cache = dataset_cache_dir / f"test_{test_size}.pt"
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
    training_params = build_training_params(model_gen, train_dataset, config, valid_dataset=valid_dataset)
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

    # ---- 5. Export results as .mat ----
    num_epochs = len(train_res["loss_train_list"])
    acc_train = train_res.get("acc_train_list", [])
    acc_valid = train_res.get("acc_valid_list", [])

    output = {
        "model_name": args.model_name,
        "config_path": args.config_path,
        # Training curves (1-D arrays for MATLAB)
        "epochs": np.arange(1, num_epochs + 1, dtype=np.float64),
        "loss_train": np.array(train_res["loss_train_list"], dtype=np.float64),
        "loss_valid": np.array(train_res["loss_valid_list"], dtype=np.float64),
        "acc_train": np.array(acc_train, dtype=np.float64) if acc_train else np.array([], dtype=np.float64),
        "acc_valid": np.array(acc_valid, dtype=np.float64) if acc_valid else np.array([], dtype=np.float64),
        # Evaluation scalars
        "test_rmspe": np.float64(test_result["loss"]),
        "test_accuracy": np.float64(test_result.get("Accuracy") or 0.0),
        "esprit_rmspe": np.float64(esprit_result["loss"]),
        # System model params
        "N": np.float64(config.system_model.N),
        "M": np.float64(config.system_model.M),
        "T": np.float64(config.system_model.T),
        "snr": np.float64(config.system_model.snr),
        "signal_nature": config.system_model.signal_nature,
        "field_type": config.system_model.field_type,
        # Training config
        "num_epochs_cfg": np.float64(config.training.epochs),
        "batch_size": np.float64(config.training.batch_size),
        "learning_rate": np.float64(config.training.learning_rate),
        "optimizer": config.training.optimizer,
        "scheduler": config.training.scheduler,
        "samples_size": np.float64(args.samples_size),
    }

    output_file = output_dir / f"{args.model_name}_results.mat"
    savemat(str(output_file), output)
    print(f"\nResults exported to: {output_file}")

    SteeringVectorGenerator.reset_instance()
    return output


if __name__ == "__main__":
    main()
