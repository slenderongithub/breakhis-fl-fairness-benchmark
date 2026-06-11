"""Run the full BreakHis federated normalization benchmark.

Trains seven federated-learning configurations on the BreakHis breast cancer
dataset (non-IID across 5 clients) and saves per-round JSON results.
"""

import torch

from config import LOCAL_EPOCHS, NUM_ROUNDS, CLASS_NAMES
from data_loader import create_non_iid_breakhis, print_client_distribution
from training_loop import SOLUTIONS, build_model, train_federated


def _select_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    """Benchmark every normalization technique under federated learning."""
    torch_device = _select_torch_device()
    print(f"Using device: {torch_device}")
    print(f"Benchmarking {len(SOLUTIONS)} solutions: {', '.join(SOLUTIONS)}")
    print(f"Rounds: {NUM_ROUNDS}, Local epochs: {LOCAL_EPOCHS}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 70)

    for solution_name in SOLUTIONS:
        print(f"\n{'='*70}")
        print(f"  Solution: {solution_name}")
        print(f"{'='*70}")

        # Determine if we need stronger augmentation (regularization solution)
        augmented = (solution_name == "regularization")

        print("Loading BreakHis dataset with non-IID partitioning...")
        train_loaders, test_loader = create_non_iid_breakhis(augmented=augmented)
        print_client_distribution(train_loaders, CLASS_NAMES)

        print("Creating model...")
        model = build_model(solution_name).to(torch_device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: BreakHisCNN (norm_type={model.norm_type}), {param_count:,} params")

        print(f"Training for {NUM_ROUNDS} rounds...")
        results = train_federated(
            train_loaders=train_loaders,
            test_loader=test_loader,
            model=model,
            solution_name=solution_name,
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
        )

        # Print final summary
        final_round = results["rounds"][-1]
        print(f"\n  Final results for {solution_name}:")
        print(f"    Accuracy:  {final_round['accuracy']:.2f}%")
        print(f"    F1 Score:  {final_round['macro_f1']:.4f}")
        print(f"    Precision: {final_round['macro_precision']:.4f}")
        print(f"    Recall:    {final_round['macro_recall']:.4f}")
        print(f"    Fairness:  {final_round['fairness_score']:.4f}")
        print(f"    Saved to:  {results.get('results_path', 'N/A')}")

    print(f"\n{'='*70}")
    print("Benchmark complete! Run analysis.py to generate comparison plots.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
