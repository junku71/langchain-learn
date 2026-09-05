from __future__ import annotations

from pathlib import Path

from ml.train_model import summarize_training_report, train_market_ensemble


def main() -> None:
    artifact = train_market_ensemble(
        save=True,
        progress=True,
    )
    summary = summarize_training_report(
        artifact,
        output_path=Path("data/ml/reports/3m_ensemble_summary.json"),
    )
    print("\n=== 3M ensemble training summary ===")
    print(f"trained_through: {summary['trained_through']}")
    print(f"horizon_days: {summary['prediction_horizon_days']}")
    print(f"train_months: {summary['train_months']}")
    print(f"validation_months: {summary['validation_months']}")
    print(f"walk_forward_step_months: {summary['walk_forward_step_months']}")
    print(f"classifier_weight: {summary['ensemble_weights']['classifier']}")
    print(f"ranker_weight: {summary['ensemble_weights']['ranker']}")
    print(f"AUC: {summary['metrics']['AUC']}")
    print(f"PR_AUC: {summary['metrics']['PR_AUC']}")
    print(f"ACC: {summary['metrics']['ACC']}")
    print(f"top15_mean_excess_return: {summary['metrics']['top15_mean_excess_return']}")
    print(f"top15_hit_rate: {summary['metrics']['top15_hit_rate']}")
    print(f"summary_path: data/ml/reports/3m_ensemble_summary.json")


if __name__ == "__main__":
    main()
