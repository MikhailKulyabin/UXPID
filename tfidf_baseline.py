"""
TF-IDF Baseline Classifier for UXPID dataset.

Provides a classical ML baseline (TF-IDF + Logistic Regression) that mirrors
the BERTTrainer interface so it can be used as a drop-in comparison in main.py.

Supports:
  - Multi-label classification  (target_field == "topics")
  - Single-label classification (branch_status, branch_type, overall_thread_sentiment)
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: build the label matrix from a DataFrame column that stores lists
# ---------------------------------------------------------------------------

def _encode_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels_column: str,
    mlb: MultiLabelBinarizer,
    is_multilabel: bool,
) -> Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer]:
    """Encode label lists into a numpy array.

    For multi-label: returns a multi-hot matrix.
    For single-label: returns a 1-D array of integer class indices.
    """
    # Collect all labels to fit the binarizer
    all_labels = []
    for row in pd.concat([train_df, test_df])[labels_column]:
        all_labels.append(row if isinstance(row, list) else [row])

    mlb.fit(all_labels)

    # Encode train
    train_lists = [
        r if isinstance(r, list) else [r]
        for r in train_df[labels_column]
    ]
    test_lists = [
        r if isinstance(r, list) else [r]
        for r in test_df[labels_column]
    ]

    y_train_bin = mlb.transform(train_lists)
    y_test_bin = mlb.transform(test_lists)

    if is_multilabel:
        return y_train_bin, y_test_bin, mlb

    # Single-label → integer class index (argmax of one-hot row)
    y_train = y_train_bin.argmax(axis=1)
    y_test = y_test_bin.argmax(axis=1)
    return y_train, y_test, mlb


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TFIDFBaseline:
    """
    TF-IDF + Logistic Regression baseline classifier for UXPID.

    Interface mirrors BERTTrainer so both can be driven from the same script.

    Parameters
    ----------
    text_field : str
        DataFrame column to use as input (``"text"`` or ``"insight_summary"``).
    target_field : str
        Label column: ``"topics"``, ``"branch_status"``, ``"branch_type"``,
        or ``"overall_thread_sentiment"``.
    max_features : int
        Maximum number of TF-IDF features.
    ngram_range : tuple
        n-gram range for TF-IDF (default: unigrams + bigrams).
    C : float
        Regularisation strength for Logistic Regression (smaller → stronger).
    max_iter : int
        Maximum iterations for the solver.
    models_output_dir : str
        Directory where the fitted pipeline is saved.
    """

    # Map target_field → labels column in the DataFrames produced by
    # ForumDataProcessor
    _LABEL_COLUMN = {
        "topics": "topics_labels",
        "branch_status": "branch_status_labels",
        "branch_type": "branch_type_labels",
        "overall_thread_sentiment": "sentiment_labels",
    }

    def __init__(
        self,
        text_field: str = "text",
        target_field: str = "topics",
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
        models_output_dir: str = ".",
        optimize_threshold: bool = True,
        support_threshold: int = 50,
    ):
        self.text_field = text_field
        self.target_field = target_field
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.models_output_dir = models_output_dir
        self.optimize_threshold = optimize_threshold
        self.support_threshold = support_threshold
        os.makedirs(self.models_output_dir, exist_ok=True)

        self.is_multilabel = target_field == "topics"

        # These are populated during train()
        self.pipeline: Optional[Pipeline] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.class_thresholds_: Optional[np.ndarray] = None  # per-class thresholds
        self._cal_fraction = 0.15  # fraction of training data held out for threshold calibration

        # Store results for reporting
        self.train_metrics: Optional[Dict[str, float]] = None
        self.val_metrics: Optional[Dict[str, float]] = None
        self.test_metrics: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _labels_column(self) -> str:
        col = self._LABEL_COLUMN.get(self.target_field)
        if col is None:
            raise ValueError(
                f"Unknown target_field '{self.target_field}'. "
                f"Must be one of {list(self._LABEL_COLUMN.keys())}."
            )
        return col

    def _build_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline."""
        tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]\w+\b",
            min_df=1,       # keep rare-class tokens
        )
        # lbfgs + L2 is fast and stable on TF-IDF sparse matrices
        lr = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
        )

        if self.is_multilabel:
            classifier = OneVsRestClassifier(lr, n_jobs=-1)
        else:
            classifier = lr  # sklearn handles multi-class natively

        return Pipeline([("tfidf", tfidf), ("clf", classifier)])

    # ------------------------------------------------------------------
    # Per-class threshold optimisation
    # ------------------------------------------------------------------

    def _find_optimal_thresholds(
        self, X_cal: List[str], y_cal_bin: np.ndarray
    ) -> np.ndarray:
        """Find per-class threshold maximising F1 on a held-out calibration set.

        Searches [0.10, 0.15, …, 0.90] for each class independently.
        Falls back to 0.5 for classes with no positive calibration examples.
        """
        from sklearn.metrics import f1_score

        y_proba = self.pipeline.predict_proba(X_cal)   # (n_samples, n_classes)
        n_classes = y_cal_bin.shape[1]
        candidate_thresholds = np.arange(0.10, 0.91, 0.05)
        best_thresholds = np.full(n_classes, 0.5)

        for i in range(n_classes):
            if y_cal_bin[:, i].sum() == 0:
                continue  # no positives → keep 0.5
            best_t, best_f1 = 0.5, -1.0
            for t in candidate_thresholds:
                y_pred_i = (y_proba[:, i] >= t).astype(int)
                f = f1_score(y_cal_bin[:, i], y_pred_i, zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            best_thresholds[i] = best_t

        return best_thresholds

    # ------------------------------------------------------------------
    # Per-class metrics table
    # ------------------------------------------------------------------

    def _compute_per_class_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> pd.DataFrame:
        """Return a DataFrame with per-class P / R / F1 / support.

        Only meaningful for multi-label targets.
        """
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        return pd.DataFrame(
            {
                "class": self.mlb.classes_,
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": sup.astype(int),
            }
        )

    def _print_paper_table(
        self,
        task_name: str,
        per_class_df: pd.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        avg_predicted: float,
        avg_true: float,
        split_name: str = "Test",
    ):
        """Print the paper-format summary table: one row for All, one for >support_threshold."""

        def _row_metrics(mask: np.ndarray, label: str):
            """Compute aggregated metrics over the classes indicated by *mask*."""
            df_sub = per_class_df[mask]
            support = int(df_sub["support"].sum())
            prec = df_sub["precision"].mean()
            rec  = df_sub["recall"].mean()
            f1   = df_sub["f1"].mean()
            # AUC over subset of classes
            indices = np.where(mask)[0]
            try:
                auc = roc_auc_score(
                    y_true[:, indices], y_proba[:, indices], average="macro"
                )
            except Exception:
                auc = float("nan")
            return label, support, prec, rec, f1, auc

        all_mask  = np.ones(len(per_class_df), dtype=bool)
        high_mask = (per_class_df["support"].values > self.support_threshold)

        rows = [_row_metrics(all_mask, "All")]
        n_high = int(high_mask.sum())
        if n_high > 0:
            rows.append(_row_metrics(high_mask, f">{ self.support_threshold} Supports"))

        hdr = (f"{'Task':<32} {'Classes':<18} {'Support':>8} "
               f"{'AVG Pred':>9} {'AVG True':>9} "
               f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
        sep = "-" * len(hdr)
        print(f"\n[{split_name}] {task_name}")
        print(sep)
        print(hdr)
        print(sep)
        for label, sup, prec, rec, f1, auc in rows:
            print(
                f"  {task_name:<30} {label:<18} {sup:>8} "
                f"{avg_predicted:>9.4f} {avg_true:>9.4f} "
                f"{prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>8.4f}"
            )
        print(sep)

        # Store high-support aggregates back as metrics
        return rows

    def _print_per_class_detail(
        self,
        per_class_df: pd.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        split_name: str = "Eval",
    ):
        """Detailed per-class table with AUC per class."""
        print(f"\n  [{split_name}] Per-class detail ({len(per_class_df)} classes)")
        print(f"  {'Class':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Support':>9}")
        print(f"  {'-'*54}")
        for i, (_, row) in enumerate(per_class_df.iterrows()):
            try:
                auc_i = roc_auc_score(y_true[:, i], y_proba[:, i])
            except Exception:
                auc_i = float("nan")
            print(
                f"  {str(row['class']):<12} "
                f"{row['precision']:>7.3f} "
                f"{row['recall']:>7.3f} "
                f"{row['f1']:>7.3f} "
                f"{auc_i:>7.3f} "
                f"{int(row['support']):>9}"
            )
        print(f"  {'-'*54}")
        print(
            f"  {'macro avg':<12} "
            f"{per_class_df['precision'].mean():>7.3f} "
            f"{per_class_df['recall'].mean():>7.3f} "
            f"{per_class_df['f1'].mean():>7.3f} "
            f"{'':>7} "
            f"{int(per_class_df['support'].sum()):>9}"
        )

    def _texts(self, df: pd.DataFrame) -> List[str]:
        return df[self.text_field].fillna("").astype(str).tolist()

    # ------------------------------------------------------------------
    # Public API (matches BERTTrainer)
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Fit the TF-IDF pipeline on *train_df* and optionally evaluate on
        *val_df* and *test_df*.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split from ForumDataProcessor.
        val_df : pd.DataFrame, optional
            Validation split (used only for reporting).
        test_df : pd.DataFrame, optional
            Test split (used only for reporting).

        Returns
        -------
        Dict with training-set metrics.
        """
        print("\n" + "=" * 60)
        print("TF-IDF BASELINE TRAINING")
        print("=" * 60)
        print(f"  Text field   : {self.text_field}")
        print(f"  Target field : {self.target_field}")
        print(f"  Classification type : {'multi-label' if self.is_multilabel else 'single-label'}")
        print(f"  TF-IDF features : {self.max_features:,}")
        print(f"  n-gram range    : {self.ngram_range}")
        print(f"  LR C            : {self.C}")
        print("=" * 60)

        labels_col = self._labels_column()

        # --- Encode labels ---
        self.mlb = MultiLabelBinarizer()
        if val_df is not None and test_df is not None:
            ref_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        elif val_df is not None:
            ref_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            ref_df = train_df.copy()

        all_label_lists = [
            r if isinstance(r, list) else [r]
            for r in ref_df[labels_col]
        ]
        self.mlb.fit(all_label_lists)
        print(f"  Classes ({len(self.mlb.classes_)}): {list(self.mlb.classes_[:10])}{'...' if len(self.mlb.classes_) > 10 else ''}")

        def encode(df: pd.DataFrame) -> np.ndarray:
            label_lists = [
                r if isinstance(r, list) else [r]
                for r in df[labels_col]
            ]
            y_bin = self.mlb.transform(label_lists)
            if self.is_multilabel:
                return y_bin
            return y_bin.argmax(axis=1)

        X_train = self._texts(train_df)
        y_train = encode(train_df)

        # --- Build and fit pipeline ---
        self.pipeline = self._build_pipeline()
        print("\nFitting TF-IDF + Logistic Regression …")

        # --- Per-class threshold calibration on a held-out slice of train ---
        # Split off a calibration set BEFORE fitting so it stays unseen by the model
        if self.is_multilabel and self.optimize_threshold:
            from sklearn.model_selection import train_test_split
            cal_size = max(50, int(len(X_train) * self._cal_fraction))
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_size,
                random_state=42,
            )
            self.pipeline.fit(X_fit, y_fit)
            print("Training complete.")
            print(f"Calibrating per-class thresholds on {len(X_cal)} held-out samples …")
            self.class_thresholds_ = self._find_optimal_thresholds(X_cal, y_cal)
            t_min = self.class_thresholds_.min()
            t_max = self.class_thresholds_.max()
            t_mean = self.class_thresholds_.mean()
            print(f"  Threshold range: [{t_min:.2f}, {t_max:.2f}]  mean={t_mean:.2f}")
        else:
            self.pipeline.fit(X_train, y_train)
            print("Training complete.")

        # --- Evaluate on splits ---
        self.train_metrics = self.evaluate(train_df, split_name="Train")
        if val_df is not None:
            self.val_metrics = self.evaluate(val_df, split_name="Validation")
        if test_df is not None:
            self.test_metrics = self.evaluate(test_df, split_name="Test")

        return self.train_metrics

    def evaluate(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        split_name: str = "Eval",
    ) -> Dict[str, float]:
        """
        Evaluate the fitted pipeline on *df*.

        Parameters
        ----------
        df : pd.DataFrame
        threshold : float
            Decision threshold for multi-label classification.
        split_name : str
            Label used in console output.

        Returns
        -------
        Dict of metric names → values (same keys as BERTTrainer.calculate_metrics).
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        labels_col = self._labels_column()
        X = self._texts(df)

        label_lists = [
            r if isinstance(r, list) else [r]
            for r in df[labels_col]
        ]
        y_true_bin = self.mlb.transform(label_lists)

        if self.is_multilabel:
            y_true = y_true_bin
            y_proba = self.pipeline.predict_proba(X)
            # Use per-class thresholds if available, else scalar threshold
            if self.class_thresholds_ is not None:
                y_pred = (y_proba >= self.class_thresholds_).astype(int)
                effective_threshold = "per-class"
            else:
                y_pred = (y_proba >= threshold).astype(int)
                effective_threshold = threshold
        else:
            y_true = y_true_bin.argmax(axis=1)
            y_pred = self.pipeline.predict(X)
            y_proba = self.pipeline.predict_proba(X)  # needed for AUC
            effective_threshold = None

        metrics = self.calculate_metrics(y_pred, y_true, threshold=threshold)

        # --- AUC ---
        try:
            if self.is_multilabel:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_proba, average="macro"
                )
            else:
                # one-vs-rest AUC for single-label
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            metrics["auc_macro"] = float("nan")

        # Build a task label for the paper table
        tf_label = "Summary" if self.text_field == "insight_summary" else "Comments"
        tgt_label = {
            "topics": "Topic",
            "overall_thread_sentiment": "Sentiment",
            "branch_status": "Branch Status",
            "branch_type": "Branch Type",
        }.get(self.target_field, self.target_field)
        task_name = f"{tf_label} → {tgt_label} (TF-IDF)"

        # --- Print aggregate summary ---
        print(f"\n[{split_name}] Metrics:")
        print(f"  F1 macro  : {metrics['f1_macro']:.4f}")
        print(f"  F1 micro  : {metrics['f1_micro']:.4f}")
        print(f"  AUC macro : {metrics['auc_macro']:.4f}")
        if self.is_multilabel:
            print(f"  Subset acc: {metrics['subset_accuracy']:.4f}")
            print(f"  Hamming   : {metrics['hamming_loss']:.4f}")
            print(f"  Jaccard   : {metrics['jaccard_score']:.4f}")
            print(f"  Threshold : {effective_threshold}")
        else:
            print(f"  Accuracy  : {metrics['accuracy']:.4f}")

        # --- Paper-format summary table and per-class detail (multi-label) ---
        if self.is_multilabel:
            per_class_df = self._compute_per_class_metrics(y_pred, y_true)
            avg_pred = float(y_pred.sum(axis=1).mean())
            avg_true = float(y_true.sum(axis=1).mean())
            table_rows = self._print_paper_table(
                task_name, per_class_df, y_true, y_proba,
                avg_pred, avg_true, split_name
            )
            self._print_per_class_detail(per_class_df, y_true, y_proba, split_name)

            # Attach per-class data to returned metrics for JSON export
            metrics["per_class"] = per_class_df.to_dict(orient="records")
            high_sup_df = per_class_df[per_class_df["support"] > self.support_threshold]
            if not high_sup_df.empty:
                high_mask = (per_class_df["support"].values > self.support_threshold)
                high_indices = np.where(high_mask)[0]
                metrics["f1_macro_high_support"] = high_sup_df["f1"].mean()
                metrics["precision_macro_high_support"] = high_sup_df["precision"].mean()
                metrics["recall_macro_high_support"] = high_sup_df["recall"].mean()
                metrics["n_high_support_classes"] = len(high_sup_df)
                metrics["support_total_high"] = int(high_sup_df["support"].sum())
                try:
                    metrics["auc_macro_high_support"] = roc_auc_score(
                        y_true[:, high_indices], y_proba[:, high_indices], average="macro"
                    )
                except Exception:
                    metrics["auc_macro_high_support"] = float("nan")
        else:
            # Single-label: print a simple paper-format row
            support = int(len(y_true))
            prec = metrics["precision_macro"]
            rec  = metrics["recall_macro"]
            f1   = metrics["f1_macro"]
            auc  = metrics["auc_macro"]
            hdr = (f"{'Task':<32} {'Classes':<10} {'Support':>8} "
                   f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
            sep = "-" * len(hdr)
            print(f"\n[{split_name}] {task_name}")
            print(sep)
            print(hdr)
            print(sep)
            print(
                f"  {task_name:<30} {'All':<10} {support:>8} "
                f"{prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>8.4f}"
            )
            print(sep)

        return metrics

    def calculate_metrics(
        self, predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics.  Signature matches BERTTrainer.calculate_metrics.
        """
        metrics: Dict[str, float] = {}

        if self.is_multilabel:
            metrics["subset_accuracy"] = accuracy_score(labels, predictions)
            metrics["hamming_loss"] = hamming_loss(labels, predictions)
            try:
                metrics["jaccard_score"] = jaccard_score(
                    labels, predictions, average="samples"
                )
            except Exception:
                metrics["jaccard_score"] = 0.0

            p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
                labels, predictions, average="macro", zero_division=0
            )
            p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
                labels, predictions, average="micro", zero_division=0
            )
            metrics.update(
                {
                    "precision_macro": p_mac,
                    "recall_macro": r_mac,
                    "f1_macro": f1_mac,
                    "precision_micro": p_mic,
                    "recall_micro": r_mic,
                    "f1_micro": f1_mic,
                    "avg_true_labels_per_sample": labels.sum(axis=1).mean(),
                    "avg_predictions_per_sample": predictions.sum(axis=1).mean(),
                    "threshold_used": threshold,
                }
            )
        else:
            metrics["accuracy"] = accuracy_score(labels, predictions)
            p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
                labels, predictions, average="macro", zero_division=0
            )
            p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
                labels, predictions, average="micro", zero_division=0
            )
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                labels, predictions, average="weighted", zero_division=0
            )
            metrics.update(
                {
                    "precision_macro": p_mac,
                    "recall_macro": r_mac,
                    "f1_macro": f1_mac,
                    "precision_micro": p_mic,
                    "recall_micro": r_mic,
                    "f1_micro": f1_mic,
                    "precision_weighted": p_w,
                    "recall_weighted": r_w,
                    "f1_weighted": f1_w,
                }
            )

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        """Save the fitted pipeline and label encoder to *filepath* (.pkl)."""
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")

        save_obj = {
            "pipeline": self.pipeline,
            "mlb": self.mlb,
            "text_field": self.text_field,
            "target_field": self.target_field,
            "is_multilabel": self.is_multilabel,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "C": self.C,
            "class_thresholds_": self.class_thresholds_,
            "support_threshold": self.support_threshold,
            "optimize_threshold": self.optimize_threshold,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(save_obj, f)
        print(f"TF-IDF model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a previously saved pipeline from *filepath*."""
        with open(filepath, "rb") as f:
            save_obj = pickle.load(f)

        self.pipeline = save_obj["pipeline"]
        self.mlb = save_obj["mlb"]
        self.text_field = save_obj["text_field"]
        self.target_field = save_obj["target_field"]
        self.is_multilabel = save_obj["is_multilabel"]
        self.max_features = save_obj.get("max_features", self.max_features)
        self.ngram_range = save_obj.get("ngram_range", self.ngram_range)
        self.C = save_obj.get("C", self.C)
        self.class_thresholds_ = save_obj.get("class_thresholds_")
        self.support_threshold = save_obj.get("support_threshold", self.support_threshold)
        self.optimize_threshold = save_obj.get("optimize_threshold", self.optimize_threshold)
        self.train_metrics = save_obj.get("train_metrics")
        self.val_metrics = save_obj.get("val_metrics")
        self.test_metrics = save_obj.get("test_metrics")
        print(f"TF-IDF model loaded from {filepath}")

    # ------------------------------------------------------------------
    # Convenience: compare with BERT results
    # ------------------------------------------------------------------

    def print_comparison(self, bert_metrics: Dict[str, float], split: str = "Test"):
        """
        Print a side-by-side metric comparison between TF-IDF and BERT.

        Parameters
        ----------
        bert_metrics : dict
            Metrics dict from BERTTrainer.calculate_metrics / BERTTrainer.evaluate.
        split : str
            Label for the header row.
        """
        tfidf_m = self.test_metrics or {}

        shared_keys = [
            "f1_macro", "f1_micro", "precision_macro", "recall_macro",
            "precision_micro", "recall_micro",
        ]
        if self.is_multilabel:
            shared_keys += ["subset_accuracy", "hamming_loss", "jaccard_score"]
        else:
            shared_keys += ["accuracy"]

        print(f"\n{'=' * 60}")
        print(f"  BASELINE COMPARISON  ({split})")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<28} {'TF-IDF':>10}  {'BERT':>10}")
        print(f"  {'-'*50}")
        for key in shared_keys:
            tval = tfidf_m.get(key, float("nan"))
            bval = bert_metrics.get(key, float("nan"))
            print(f"  {key:<28} {tval:>10.4f}  {bval:>10.4f}")
        print(f"{'=' * 60}\n")

    def save_comparison(
        self,
        bert_metrics: Dict[str, float],
        save_path: str,
        split: str = "Test",
    ):
        """
        Save a JSON file with TF-IDF and BERT metrics side by side.
        """
        comparison = {
            "split": split,
            "tfidf": self.test_metrics or {},
            "bert": bert_metrics,
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to {save_path}")

    def plot_comparison(
        self,
        bert_metrics: Dict[str, float],
        save_path: str = "tfidf_vs_bert.png",
        split: str = "Test",
    ):
        """
        Bar chart comparing TF-IDF and BERT on key metrics.
        """
        tfidf_m = self.test_metrics or {}

        keys = ["f1_macro", "f1_micro", "precision_macro", "recall_macro"]
        if not self.is_multilabel:
            keys.insert(0, "accuracy")

        labels = [k.replace("_", "\n") for k in keys]
        tfidf_vals = [tfidf_m.get(k, 0.0) for k in keys]
        bert_vals = [bert_metrics.get(k, 0.0) for k in keys]

        x = np.arange(len(keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width / 2, tfidf_vals, width, label="TF-IDF + LR", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, bert_vals, width, label="BERT", color="#DD8452")

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(f"TF-IDF Baseline vs BERT  ({split})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
        ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison plot saved to {save_path}")

    # ------------------------------------------------------------------
    # Predict helpers (mirrors BERTPredictor interface)
    # ------------------------------------------------------------------

    def predict_single(self, text: str, threshold: float = 0.5) -> Dict:
        """Predict labels for a single text string."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        if self.is_multilabel:
            proba = self.pipeline.predict_proba([text])[0]
            binary = (proba >= threshold).astype(int)
            predicted_classes = self.mlb.classes_[binary.astype(bool)]
            scores = {cls: float(proba[i]) for i, cls in enumerate(self.mlb.classes_)}
            return {
                "predicted_labels": list(predicted_classes),
                "scores": scores,
                "threshold": threshold,
            }
        else:
            pred = self.pipeline.predict([text])[0]
            label = self.mlb.classes_[pred]
            proba = self.pipeline.predict_proba([text])[0]
            scores = {cls: float(proba[i]) for i, cls in enumerate(self.mlb.classes_)}
            return {
                "predicted_label": label,
                "scores": scores,
            }

    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """Predict labels for a list of texts."""
        return [self.predict_single(t, threshold) for t in texts]


# ---------------------------------------------------------------------------
# Standalone pipeline (mirrors main.py)
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str, timestamp: str):
    """Configure root logger to file + stdout."""
    import logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"tfidf_baseline_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialised. Log file: {log_file}")
    return logger


def main():
    """Full standalone TF-IDF baseline pipeline.

    Run:
        python tfidf_baseline.py
        python tfidf_baseline.py --target-field branch_status --text-field insight_summary
    """
    import argparse
    import ast
    import logging
    import sys
    from datetime import datetime

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from data_processor import ForumDataProcessor
    from config import (
        DATA_CONFIG, TRAINING_CONFIG, OUTPUT_CONFIG,
        LOGGING_CONFIG, TOPICS_CONFIG,
    )
    from utils import create_timestamp_folder, get_timestamp_paths, create_run_summary

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="TF-IDF + Logistic Regression baseline for UXPID"
    )
    parser.add_argument("--dataset-path", default=DATA_CONFIG["dataset_path"],
                        help="Path to raw dataset directory")
    parser.add_argument("--output-dir", default=OUTPUT_CONFIG["results_dir"],
                        help="Directory to save results")
    parser.add_argument("--text-field", choices=["text", "insight_summary"],
                        default=DATA_CONFIG["text_field"],
                        help="Input field to use for training")
    parser.add_argument("--target-field",
                        choices=["topics", "branch_status", "branch_type", "overall_thread_sentiment"],
                        default=DATA_CONFIG["target_field"],
                        help="Target label field")
    parser.add_argument("--data-split", action="store_true",
                        default=DATA_CONFIG["data_split"],
                        help="Create a new train/test split (ignores existing processed data)")
    parser.add_argument("--use-official-split", action="store_true", default=False,
                        help="Use the official Zenodo split from splits/ instead of a random split (requires --data-split)")
    parser.add_argument("--splits-dir", type=str, default=DATA_CONFIG.get("splits_dir", "splits"),
                        help="Path to directory containing train_branches.txt / test_branches.txt")
    parser.add_argument("--skip-data-processing", action="store_true",
                        help="Load existing processed_data/ without re-processing")
    parser.add_argument("--max-features", type=int, default=50_000,
                        help="Maximum TF-IDF vocabulary size")
    parser.add_argument("--ngram-max", type=int, default=2,
                        help="Upper bound of n-gram range (1 = unigrams only, 2 = unigrams+bigrams)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic Regression regularisation strength")
    parser.add_argument("--optimize-threshold", action="store_true", default=True,
                        help="Optimise per-class decision threshold on training data (multi-label only)")
    parser.add_argument("--no-optimize-threshold", dest="optimize_threshold", action="store_false",
                        help="Use a fixed 0.5 threshold instead of per-class optimisation")
    parser.add_argument("--support-threshold", type=int, default=50,
                        help="Min test-set support to include a class in the high-support metrics table")
    parser.add_argument("--use-timestamp-folder", action="store_true",
                        default=OUTPUT_CONFIG["use_timestamp_folders"],
                        help="Organise outputs in a timestamped sub-folder")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.use_timestamp_folder:
        _, output_base_path = create_timestamp_folder(OUTPUT_CONFIG["outputs_base_dir"])
        paths = get_timestamp_paths(output_base_path)
        results_dir = paths["results"]
        log_dir     = paths["logs"]
        plots_dir   = paths["plots"]
        models_dir  = paths["models"]
        print(f"Outputs will be saved under: {output_base_path}")
    else:
        output_base_path = args.output_dir
        results_dir = args.output_dir
        log_dir     = OUTPUT_CONFIG["logs_dir"]
        plots_dir   = os.path.join(args.output_dir, OUTPUT_CONFIG["plots_dir"])
        models_dir  = args.output_dir
        os.makedirs(results_dir, exist_ok=True)

    logger = _setup_logging(log_dir, timestamp)
    logger.info("=" * 60)
    logger.info("TF-IDF BASELINE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Text field   : {args.text_field}")
    logger.info(f"Target field : {args.target_field}")
    logger.info(f"Max features : {args.max_features:,}")
    logger.info(f"n-gram range : (1, {args.ngram_max})")
    logger.info(f"LR C         : {args.C}")

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("STEP 1: DATA")
    logger.info("=" * 50)

    processed_data_dir = DATA_CONFIG["processed_data_dir"]
    processor = ForumDataProcessor(
        args.dataset_path,
        topics_file=TOPICS_CONFIG["topics_file"],
        target_field=args.target_field,
    )

    if args.skip_data_processing and not args.data_split and os.path.exists(processed_data_dir):
        logger.info("Loading existing processed data ...")
        train_df, test_df = processor.load_processed_data(processed_data_dir)
    else:
        data = processor.load_data()
        if not data:
            logger.error("No data loaded. Check --dataset-path.")
            return
        processor.create_dataframe()
        train_df, test_df = processor.get_or_create_training_data(
            args.data_split,
            splits_dir=args.splits_dir if args.use_official_split else None
        )

    if train_df is None or test_df is None:
        logger.error("Could not obtain train/test split. "
                     "Run with --data-split to create one.")
        return

    # Convert any string-encoded list columns back to Python lists
    label_cols = ["topics_labels", "branch_status_labels",
                  "branch_type_labels", "sentiment_labels"]
    for col in label_cols:
        for df in (train_df, test_df):
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(ast.literal_eval)

    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # ------------------------------------------------------------------
    # Step 2: Train
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("STEP 2: TRAINING")
    logger.info("=" * 50)

    baseline = TFIDFBaseline(
        text_field=args.text_field,
        target_field=args.target_field,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        C=args.C,
        models_output_dir=models_dir,
        optimize_threshold=args.optimize_threshold,
        support_threshold=args.support_threshold,
    )

    baseline.train(train_df, test_df=test_df)

    # Save fitted model
    model_path = os.path.join(models_dir, "tfidf_baseline.pkl")
    baseline.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # Step 3: Results
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("STEP 3: RESULTS")
    logger.info("=" * 50)

    test_metrics = baseline.test_metrics or {}

    # Save metrics JSON
    results = {
        "text_field": args.text_field,
        "target_field": args.target_field,
        "model": "TF-IDF + Logistic Regression",
        "max_features": args.max_features,
        "ngram_range": [1, args.ngram_max],
        "C": args.C,
        "train_metrics": baseline.train_metrics,
        "test_metrics": test_metrics,
    }
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_file = os.path.join(results_dir, "tfidf_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    logger.info(f"Results saved to: {results_file}")

    # Save comparison plot (TF-IDF only — no BERT metrics yet)
    try:
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "tfidf_metrics.png")

        keys = ["f1_macro", "f1_micro", "precision_macro", "recall_macro"]
        if not baseline.is_multilabel:
            keys.insert(0, "accuracy")
        labels_x = [k.replace("_", "\n") for k in keys]
        values   = [test_metrics.get(k, 0.0) for k in keys]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels_x, values, color="#4C72B0")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"TF-IDF Baseline – {args.target_field} ({args.text_field})")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Metrics plot saved to: {plot_path}")
    except Exception as exc:
        logger.warning(f"Could not create metrics plot: {exc}")

    if args.use_timestamp_folder:
        create_run_summary(output_base_path, results, timestamp)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("BASELINE RUN COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Target field : {args.target_field}")
    logger.info(f"  Text field   : {args.text_field}")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples : {len(test_df)}")
    for k, v in test_metrics.items():
        if k == "per_class":
            continue  # printed inline during evaluate(); too verbose for log summary
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if "f1_macro_high_support" in test_metrics:
        logger.info(
            f"  f1_macro (support>{args.support_threshold}): "
            f"{test_metrics['f1_macro_high_support']:.4f}  "
            f"({int(test_metrics['n_high_support_classes'])} classes)"
        )
    logger.info(f"  Results dir  : {results_dir}")


if __name__ == "__main__":
    main()
