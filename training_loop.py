"""Federated training loop with per-client resource measurements.

Benchmarks seven normalization/loss strategies on BreakHis breast cancer data:
    baseline, batch_norm, group_norm, layer_norm, instance_norm, focal_loss, regularization
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    CLIENT_LEARNING_RATE,
    NUM_CLASSES,
    PROJECT_ROOT,
)
from models.cnn import BreakHisCNN, ClientUpdate, fedavg
from solutions import SOLUTION_REGISTRY
from solutions.focal_loss import FocalLoss


# ---------------------------------------------------------------------------
# Solution names (ordered)
# ---------------------------------------------------------------------------

SOLUTIONS = list(SOLUTION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_model(solution_name: str) -> BreakHisCNN:
    """Create a BreakHisCNN with the appropriate normalization for *solution_name*."""
    sol = SOLUTION_REGISTRY.get(solution_name)
    if sol is None:
        raise ValueError(f"Unknown solution: {solution_name!r}. Must be one of {SOLUTIONS}")
    return BreakHisCNN(num_classes=NUM_CLASSES, norm_type=sol.NORM_TYPE)


def compute_class_weights(train_loaders, num_classes=NUM_CLASSES, device="cpu"):
    """Compute inverse-frequency class weights from all training loaders.

    This balances the loss so the model doesn't collapse to predicting
    only the majority class (malignant ≈ 69% of BreakHis).
    """
    counts = torch.zeros(num_classes)
    for loader in train_loaders:
        for _, targets in loader:
            for t in targets:
                counts[t.item()] += 1

    # Inverse frequency: more weight to minority class
    total = counts.sum()
    weights = total / (num_classes * counts)
    # Clamp to avoid extreme weights
    weights = torch.clamp(weights, min=0.5, max=3.0)
    return weights.to(device)


def _build_criterion(solution_name, class_weights=None):
    """Build the loss function for a given solution."""
    sol = SOLUTION_REGISTRY[solution_name]
    if sol.LOSS_TYPE == "focal":
        # Pass per-class weights as alpha so FocalLoss handles class imbalance
        return FocalLoss(alpha=class_weights, gamma=2.0)
    # Standard CE with class weights to handle imbalance
    return nn.CrossEntropyLoss(weight=class_weights)


def _build_optimizer(model, solution_name):
    """Build the optimizer for a given solution."""
    sol = SOLUTION_REGISTRY[solution_name]
    weight_decay = sol.WEIGHT_DECAY
    return torch.optim.Adam(
        model.parameters(),
        lr=CLIENT_LEARNING_RATE,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Client training
# ---------------------------------------------------------------------------


def _train_client(client_model, train_loader, criterion, solution_name, local_epochs, device):
    """Train a single client and return its updated state dict + stats."""
    client_model.train()
    optimizer = _build_optimizer(client_model, solution_name)
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    total_correct = 0

    start_time = time.time()

    for _ in range(local_epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = client_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            # Gradient clipping to prevent client drift with non-IID data
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_batches += 1

    elapsed = time.time() - start_time

    stats = {
        "avg_loss": total_loss / total_samples if total_samples else 0.0,
        "train_accuracy": total_correct / total_samples if total_samples else 0.0,
        "num_batches": total_batches,
        "num_samples": total_samples,
        "training_time_sec": elapsed,
    }
    return client_model.state_dict(), stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_global_model(model, test_loader, device, num_classes=NUM_CLASSES):
    """Evaluate accuracy, per-class accuracy, precision, recall, F1, and fairness."""
    model.eval()
    class_tp = [0] * num_classes     # true positives
    class_total = [0] * num_classes  # actual count per class
    class_predicted = [0] * num_classes  # predicted count per class

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs).argmax(dim=1)

            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            for class_id in range(num_classes):
                actual_mask = targets == class_id
                pred_mask = predictions == class_id
                class_total[class_id] += actual_mask.sum().item()
                class_predicted[class_id] += pred_mask.sum().item()
                # TP: predicted as class_id AND actually class_id
                class_tp[class_id] += (pred_mask & actual_mask).sum().item()

    accuracy = (total_correct / total_samples) * 100 if total_samples else 0.0

    class_accuracy = {}
    precision = {}
    recall = {}
    f1 = {}

    for class_id in range(num_classes):
        # Per-class accuracy (recall per class)
        acc = (
            (class_tp[class_id] / class_total[class_id]) * 100
            if class_total[class_id]
            else 0.0
        )
        class_accuracy[str(class_id)] = acc

        # Precision: TP / (TP + FP) = TP / total predicted as class
        prec = (
            class_tp[class_id] / class_predicted[class_id]
            if class_predicted[class_id]
            else 0.0
        )
        precision[str(class_id)] = prec

        # Recall: TP / (TP + FN) = TP / total actual class
        rec = (
            class_tp[class_id] / class_total[class_id]
            if class_total[class_id]
            else 0.0
        )
        recall[str(class_id)] = rec

        # F1
        f1_val = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else 0.0
        )
        f1[str(class_id)] = f1_val

    # Fairness: 1 - (max_class_acc - min_class_acc) / 100
    observed_class_acc = [
        acc_val
        for cid, acc_val in class_accuracy.items()
        if class_total[int(cid)] > 0
    ]
    if observed_class_acc:
        fairness_score = 1 - (max(observed_class_acc) - min(observed_class_acc)) / 100
    else:
        fairness_score = 0.0

    # Macro-averaged metrics
    macro_precision = sum(precision.values()) / num_classes
    macro_recall = sum(recall.values()) / num_classes
    macro_f1 = sum(f1.values()) / num_classes

    return {
        "accuracy": accuracy,
        "fairness_score": fairness_score,
        "class_accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def _save_results(results, solution_name):
    results_dir = Path(PROJECT_ROOT) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{solution_name}_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Main federated training loop
# ---------------------------------------------------------------------------


def train_federated(
    train_loaders,
    test_loader,
    model,
    solution_name,
    num_rounds=20,
    local_epochs=2,
):
    """Train a model with FedAvg and collect per-round metrics.

    Parameters
    ----------
    train_loaders : list[DataLoader]
        One DataLoader per client.
    test_loader : DataLoader
        Global test set DataLoader.
    model : nn.Module
        The global model (will be updated in-place).
    solution_name : str
        Name of the normalization/loss technique being benchmarked.
    num_rounds : int
        Number of federated communication rounds.
    local_epochs : int
        Number of local SGD epochs per client per round.

    Returns
    -------
    dict
        Full results including per-round metrics.
    """
    if solution_name not in SOLUTIONS:
        raise ValueError(f"solution_name must be one of: {sorted(SOLUTIONS)}")

    device = _get_device(model)

    # Compute class weights once from training data to handle class imbalance
    class_weights = compute_class_weights(train_loaders, device=device)
    print(f"  Class weights: benign={class_weights[0]:.3f}, malignant={class_weights[1]:.3f}")

    criterion = _build_criterion(solution_name, class_weights=class_weights)

    # Define checkpoint path
    checkpoint_path = Path(PROJECT_ROOT) / "results" / f"{solution_name}_checkpoint.pth"
    start_round = 0
    results = None

    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if checkpoint.get("round", 0) < num_rounds:
                print(f"  [Checkpoint] Found existing checkpoint. Resuming from round {checkpoint['round'] + 1}...")
                model.load_state_dict(checkpoint["model_state_dict"])
                results = checkpoint["results"]
                start_round = checkpoint["round"]
        except Exception as e:
            print(f"  [Checkpoint Warning] Failed to load checkpoint: {e}. Starting from scratch.")

    if results is None:
        sol = SOLUTION_REGISTRY[solution_name]
        results = {
            "solution_name": solution_name,
            "norm_type": sol.NORM_TYPE,
            "loss_type": sol.LOSS_TYPE,
            "description": sol.DESCRIPTION,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "num_clients": len(train_loaders),
            "rounds": [],
        }

    for round_index in range(start_round, num_rounds):
        round_start = time.time()
        client_state_dicts = []
        client_stats = []

        for client_id, train_loader in enumerate(train_loaders):
            client_model = copy.deepcopy(model).to(device)
            client_state, stats = _train_client(
                client_model=client_model,
                train_loader=train_loader,
                criterion=criterion,
                solution_name=solution_name,
                local_epochs=local_epochs,
                device=device,
            )
            stats["client_id"] = client_id
            client_state_dicts.append(client_state)
            client_stats.append(stats)

        # Aggregate via FedAvg
        client_updates = [
            ClientUpdate(
                client_id=stats["client_id"],
                state_dict=state_dict,
                num_samples=stats["num_samples"],
                train_loss=stats["avg_loss"],
                train_accuracy=stats["train_accuracy"],
            )
            for state_dict, stats in zip(client_state_dicts, client_stats)
        ]
        model.load_state_dict(fedavg(client_updates))

        # Recalibrate BatchNorm running statistics after aggregation.
        # Averaged running stats from non-IID clients are meaningless;
        # a quick forward pass recomputes them from representative data.
        model.calibrate_bn(train_loaders, device, max_batches=5)

        # Evaluate global model
        eval_metrics = _evaluate_global_model(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        avg_round_loss = (
            sum(s["avg_loss"] for s in client_stats) / len(client_stats)
            if client_stats else 0.0
        )
        total_round_time = time.time() - round_start

        round_result = {
            "round": round_index + 1,
            "accuracy": eval_metrics["accuracy"],
            "fairness_score": eval_metrics["fairness_score"],
            "avg_train_loss": avg_round_loss,
            "class_accuracy": eval_metrics["class_accuracy"],
            "macro_precision": eval_metrics["macro_precision"],
            "macro_recall": eval_metrics["macro_recall"],
            "macro_f1": eval_metrics["macro_f1"],
            "round_time_sec": total_round_time,
            "clients": client_stats,
        }
        results["rounds"].append(round_result)
        results["results_path"] = str(_save_results(results, solution_name))

        # Save checkpoint to disk
        checkpoint = {
            "round": round_index + 1,
            "model_state_dict": model.state_dict(),
            "results": results
        }
        torch.save(checkpoint, checkpoint_path)

        print(
            f"  [{solution_name}] Round {round_index + 1}/{num_rounds}: "
            f"loss={avg_round_loss:.4f}, "
            f"acc={eval_metrics['accuracy']:.2f}%, "
            f"F1={eval_metrics['macro_f1']:.3f}, "
            f"fair={eval_metrics['fairness_score']:.3f}, "
            f"time={total_round_time:.1f}s"
        )

    # Clean up checkpoint file on successful completion of all rounds
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    return results
