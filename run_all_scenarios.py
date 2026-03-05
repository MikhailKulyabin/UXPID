"""
Run all four validation scenarios for the TF-IDF baseline.

Scenarios:
  1. Insights summary  -> Topic class          (multi-label, 32 classes)
  2. Raw comments      -> Topic class          (multi-label, 32 classes)
  3. Insights summary  -> Overall Sentiment    (single-label, 3 classes)
  4. Raw comments      -> Overall Sentiment    (single-label, 3 classes)

Usage:
    python run_all_scenarios.py                      # use existing processed data
    python run_all_scenarios.py --data-split         # force fresh split
    python run_all_scenarios.py --no-official-split  # random split instead of Zenodo
"""

import os
import sys
import ast
import json
import argparse
import logging
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_data(processor, args, target_field, logger):
    """Load or create train/test split for *target_field*."""
    from config import DATA_CONFIG

    processed_data_dir = DATA_CONFIG["processed_data_dir"]
    processor.target_field = target_field

    has_processed = os.path.exists(
        os.path.join(processed_data_dir, f"label_encoder_{target_field}.pkl")
    )

    if not args.data_split and has_processed:
        logger.info(f"Loading existing processed data for '{target_field}' ...")
        train_df, test_df = processor.load_processed_data(processed_data_dir)
    else:
        logger.info(f"Processing data for '{target_field}' ...")
        if not getattr(processor, "_raw_loaded", False):
            processor.load_data()
            processor._raw_loaded = True
        processor.create_dataframe()
        splits_dir = args.splits_dir if not args.no_official_split else None
        train_df, test_df = processor.get_or_create_training_data(
            data_split=True, splits_dir=splits_dir,
        )

    # Convert string-encoded list columns back to Python lists
    label_cols = [
        "topics_labels", "branch_status_labels",
        "branch_type_labels", "sentiment_labels",
    ]
    for col in label_cols:
        for df in (train_df, test_df):
            if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], str):
                try:
                    df[col] = df[col].apply(ast.literal_eval)
                except Exception:
                    pass

    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all four TF-IDF baseline scenarios for UXPID validation",
    )
    parser.add_argument("--data-split", action="store_true",
                        help="Force new data split (required on first run per target)")
    parser.add_argument("--no-official-split", action="store_true",
                        help="Use random split instead of official Zenodo split")
    parser.add_argument("--splits-dir", default="splits",
                        help="Path to train/test split txt files")
    parser.add_argument("--support-threshold", type=int, default=50,
                        help="Min support to include a class in the high-support subset")
    parser.add_argument("--max-features", type=int, default=50_000,
                        help="Maximum TF-IDF vocabulary size")
    parser.add_argument("--ngram-max", type=int, default=2,
                        help="Upper bound of n-gram range")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic Regression regularisation strength")
    parser.add_argument("--output-json", default="all_scenarios_results.json",
                        help="File to save the consolidated results JSON")
    args = parser.parse_args()

    logger = _setup_logging()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tfidf_baseline import TFIDFBaseline
    from data_processor import ForumDataProcessor
    from config import DATA_CONFIG, TOPICS_CONFIG

    processor = ForumDataProcessor(
        DATA_CONFIG["dataset_path"],
        topics_file=TOPICS_CONFIG["topics_file"],
        target_field="topics",
    )

    # Scenario definitions
    SCENARIOS = [
        ("topics",                   "insight_summary"),
        ("topics",                   "text"),
        ("overall_thread_sentiment", "insight_summary"),
        ("overall_thread_sentiment", "text"),
    ]

    all_results = {}
    last_target = None
    train_df = test_df = None

    for target_field, text_field in SCENARIOS:
        task_label = (
            ("Summary" if text_field == "insight_summary" else "Comments")
            + (" -> Topic" if target_field == "topics" else " -> Sentiment")
        )
        logger.info("=" * 60)
        logger.info(f"SCENARIO: {task_label}  text={text_field}  target={target_field}")
        logger.info("=" * 60)

        # Only reload data when target changes
        if target_field != last_target:
            train_df, test_df = _load_data(processor, args, target_field, logger)
            last_target = target_field

        baseline = TFIDFBaseline(
            text_field=text_field,
            target_field=target_field,
            max_features=args.max_features,
            ngram_range=(1, args.ngram_max),
            C=args.C,
            models_output_dir=".",
            optimize_threshold=(target_field == "topics"),
            support_threshold=args.support_threshold,
        )

        baseline.train(train_df, test_df=test_df)
        m = baseline.test_metrics or {}

        # Collect summary metrics
        scenario_key = f"{text_field}__{target_field}"
        summary = {
            "task": task_label,
            "text_field": text_field,
            "target_field": target_field,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "f1_macro": m.get("f1_macro"),
            "f1_micro": m.get("f1_micro"),
            "precision_macro": m.get("precision_macro"),
            "recall_macro": m.get("recall_macro"),
            "auc_macro": m.get("auc_macro"),
        }
        if target_field == "topics":
            summary["avg_predictions_per_sample"] = m.get("avg_predictions_per_sample")
            summary["avg_true_labels_per_sample"] = m.get("avg_true_labels_per_sample")
            summary["f1_macro_high_support"] = m.get("f1_macro_high_support")
            summary["auc_macro_high_support"] = m.get("auc_macro_high_support")
            summary["n_high_support_classes"] = m.get("n_high_support_classes")
        else:
            summary["accuracy"] = m.get("accuracy")

        all_results[scenario_key] = summary

    # ------------------------------------------------------------------
    # Save consolidated JSON
    # ------------------------------------------------------------------
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"Consolidated results saved to: {args.output_json}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  ALL SCENARIOS — Test Metrics Summary")
    print("=" * 90)
    header = (
        f"  {'Task':<30} {'F1 macro':>9} {'F1 micro':>9} "
        f"{'Prec':>7} {'Rec':>7} {'AUC':>7}"
    )
    print(header)
    print("  " + "-" * 72)
    for key, s in all_results.items():
        print(
            f"  {s['task']:<30} "
            f"{s['f1_macro']:>9.4f} {s['f1_micro']:>9.4f} "
            f"{s['precision_macro']:>7.4f} {s['recall_macro']:>7.4f} "
            f"{s['auc_macro']:>7.4f}"
        )
        if s.get("f1_macro_high_support") is not None:
            print(
                f"    (>{args.support_threshold} support: "
                f"F1={s['f1_macro_high_support']:.4f}  "
                f"AUC={s['auc_macro_high_support']:.4f}  "
                f"{s['n_high_support_classes']} classes)"
            )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
