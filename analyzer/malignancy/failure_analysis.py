import logging
import os
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import entropy
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from shared_lib.utils.utils import print_config, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MalignancyFailureAnalyzer:
    """Comprehensive failure analysis for malignancy models"""

    def __init__(self, df: pd.DataFrame, model_cols: List[str], ensemble_col: str = "prob_ensemble"):
        self.df = df
        self.model_cols = model_cols
        self.ensemble_col = ensemble_col
        self.y_true = df["annotation"]

    def analyze_basic_metrics(self) -> Dict:
        """Calculate basic performance metrics for all models"""
        metrics = {}

        for col in self.model_cols + [self.ensemble_col]:
            # AUROC
            auroc = roc_auc_score(self.y_true, self.df[col])

            # ROC curve for threshold analysis
            fpr, tpr, thresholds = roc_curve(self.y_true, self.df[col])

            # Youden's J statistic
            J = tpr - fpr
            youden_idx = np.argmax(J)
            youden_threshold = thresholds[youden_idx]

            # F1 score at different thresholds
            f1_scores = []
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred = (self.df[col] > threshold).astype(int)
                f1 = f1_score(self.y_true, y_pred)
                f1_scores.append(f1)

            best_f1_idx = np.argmax(f1_scores)
            best_f1_threshold = 0.1 + best_f1_idx * 0.05

            # Sensitivity at 95% specificity
            specificity_95_idx = np.where(1 - fpr >= 0.95)[0]
            sensitivity_95 = max(tpr[specificity_95_idx]) if len(specificity_95_idx) > 0 else 0.0

            # Specificity at 95% sensitivity
            sensitivity_95_idx = np.where(tpr >= 0.95)[0]
            specificity_95 = max(1 - fpr[sensitivity_95_idx]) if len(sensitivity_95_idx) > 0 else 0.0

            metrics[col] = {
                "auroc": auroc,
                "youden_threshold": youden_threshold,
                "best_f1_threshold": best_f1_threshold,
                "sensitivity_95_specificity": sensitivity_95,
                "specificity_95_sensitivity": specificity_95,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            }

        return metrics

    def identify_failure_cases(self, threshold: float = 0.5) -> Dict:
        """Identify different types of failure cases"""
        failures = {
            "false_positives": [],
            "false_negatives": [],
            "high_confidence_errors": [],
            "low_confidence_correct": [],
            "ensemble_disagreement": [],
        }

        # Calculate ensemble variance
        self.df["ensemble_variance"] = self.df[self.model_cols].var(axis=1)
        self.df["ensemble_entropy"] = self.df[self.model_cols].apply(
            lambda x: entropy([x.mean(), 1 - x.mean()]), axis=1
        )

        for idx, row in self.df.iterrows():
            y_true = float(row["annotation"])
            y_pred_ensemble = 1 if row[self.ensemble_col] > threshold else 0

            # False positives
            if y_true == 0.0 and y_pred_ensemble == 1:
                failures["false_positives"].append(
                    {
                        "idx": idx,
                        "prob_ensemble": row[self.ensemble_col],
                        "probs_individual": {col: row[col] for col in self.model_cols},
                        "variance": row["ensemble_variance"],
                        "entropy": row["ensemble_entropy"],
                    }
                )

            # False negatives
            elif y_true == 1.0 and y_pred_ensemble == 0:
                failures["false_negatives"].append(
                    {
                        "idx": idx,
                        "prob_ensemble": row[self.ensemble_col],
                        "probs_individual": {col: row[col] for col in self.model_cols},
                        "variance": row["ensemble_variance"],
                        "entropy": row["ensemble_entropy"],
                    }
                )

            # High confidence errors (ensemble prob > 0.8 for wrong prediction)
            if (y_true == 0.0 and row[self.ensemble_col] > 0.8) or (y_true == 1.0 and row[self.ensemble_col] < 0.2):
                failures["high_confidence_errors"].append(
                    {
                        "idx": idx,
                        "prob_ensemble": row[self.ensemble_col],
                        "probs_individual": {col: row[col] for col in self.model_cols},
                        "variance": row["ensemble_variance"],
                        "entropy": row["ensemble_entropy"],
                    }
                )

            # Low confidence correct (ensemble prob between 0.4-0.6 for correct prediction)
            if (y_true == 0.0 and 0.4 <= row[self.ensemble_col] <= 0.6) or (
                y_true == 1.0 and 0.4 <= row[self.ensemble_col] <= 0.6
            ):
                failures["low_confidence_correct"].append(
                    {
                        "idx": idx,
                        "prob_ensemble": row[self.ensemble_col],
                        "probs_individual": {col: row[col] for col in self.model_cols},
                        "variance": row["ensemble_variance"],
                        "entropy": row["ensemble_entropy"],
                    }
                )

            # High ensemble disagreement (variance > 0.01)
            if row["ensemble_variance"] > 0.01:
                failures["ensemble_disagreement"].append(
                    {
                        "idx": idx,
                        "prob_ensemble": row[self.ensemble_col],
                        "probs_individual": {col: row[col] for col in self.model_cols},
                        "variance": row["ensemble_variance"],
                        "entropy": row["ensemble_entropy"],
                    }
                )

        return failures

    def analyze_ensemble_methods(self) -> Dict:
        """Compare different ensemble methods"""
        ensemble_methods = {}

        # Current average ensemble
        ensemble_methods["average"] = {
            "probs": self.df[self.ensemble_col],
            "auroc": roc_auc_score(self.y_true, self.df[self.ensemble_col]),
        }

        # Weighted average (based on individual AUROC)
        individual_aurocs = [roc_auc_score(self.y_true, self.df[col]) for col in self.model_cols]
        weights = np.array(individual_aurocs) / sum(individual_aurocs)
        weighted_probs = np.zeros(len(self.df))
        for i, col in enumerate(self.model_cols):
            weighted_probs += weights[i] * self.df[col].values
        ensemble_methods["weighted"] = {
            "probs": weighted_probs,
            "auroc": roc_auc_score(self.y_true, weighted_probs),
            "weights": weights,
        }

        # Geometric mean
        geometric_probs = np.exp(np.mean(np.log(self.df[self.model_cols] + 1e-10), axis=1))
        ensemble_methods["geometric"] = {"probs": geometric_probs, "auroc": roc_auc_score(self.y_true, geometric_probs)}

        # Median
        median_probs = self.df[self.model_cols].median(axis=1)
        ensemble_methods["median"] = {"probs": median_probs, "auroc": roc_auc_score(self.y_true, median_probs)}

        # Max probability
        max_probs = self.df[self.model_cols].max(axis=1)
        ensemble_methods["max"] = {"probs": max_probs, "auroc": roc_auc_score(self.y_true, max_probs)}

        # Min probability
        min_probs = self.df[self.model_cols].min(axis=1)
        ensemble_methods["min"] = {"probs": min_probs, "auroc": roc_auc_score(self.y_true, min_probs)}

        return ensemble_methods

    def optimize_thresholds(self) -> Dict:
        """Find optimal thresholds for different metrics"""
        thresholds = {}

        for col in self.model_cols + [self.ensemble_col]:
            fpr, tpr, thresh = roc_curve(self.y_true, self.df[col])

            # Youden's J statistic
            J = tpr - fpr
            youden_idx = np.argmax(J)
            thresholds[f"{col}_youden"] = thresh[youden_idx]

            # F1 score optimization
            best_f1 = 0
            best_thresh = 0.5
            for t in np.arange(0.1, 0.9, 0.01):
                y_pred = (self.df[col] > t).astype(int)
                f1 = f1_score(self.y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
            thresholds[f"{col}_f1"] = best_thresh

            # Sensitivity at 95% specificity
            specificity_95_idx = np.where(1 - fpr >= 0.95)[0]
            if len(specificity_95_idx) > 0:
                thresholds[f"{col}_sens95"] = thresh[specificity_95_idx[0]]

            # Specificity at 95% sensitivity
            sensitivity_95_idx = np.where(tpr >= 0.95)[0]
            if len(sensitivity_95_idx) > 0:
                thresholds[f"{col}_spec95"] = thresh[sensitivity_95_idx[0]]

        return thresholds

    def create_visualizations(self, output_dir: str):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. ROC curves comparison
        plt.figure(figsize=(12, 8))
        for col in self.model_cols + [self.ensemble_col]:
            fpr, tpr, _ = roc_curve(self.y_true, self.df[col])
            auroc = roc_auc_score(self.y_true, self.df[col])
            plt.plot(fpr, tpr, label=f"{col} (AUROC = {auroc:.4f})", linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison - Malignancy Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Probability distributions
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, col in enumerate(self.model_cols + [self.ensemble_col]):
            if i < len(axes):
                axes[i].hist(self.df[self.df["annotation"] == 0][col], bins=50, alpha=0.7, label="Benign", density=True)
                axes[i].hist(
                    self.df[self.df["annotation"] == 1][col], bins=50, alpha=0.7, label="Malignant", density=True
                )
                axes[i].set_title(f"{col} Distribution")
                axes[i].set_xlabel("Probability")
                axes[i].set_ylabel("Density")
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/probability_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Ensemble variance analysis
        plt.figure(figsize=(12, 8))
        plt.scatter(
            self.df["ensemble_variance"], self.df[self.ensemble_col], c=self.df["annotation"], alpha=0.6, cmap="viridis"
        )
        plt.xlabel("Ensemble Variance")
        plt.ylabel("Ensemble Probability")
        plt.title("Ensemble Variance vs Probability - Malignancy")
        plt.colorbar(label="True Label (0: Benign, 1: Malignant)")
        plt.savefig(f"{output_dir}/ensemble_variance.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Model correlation heatmap
        corr_matrix = self.df[self.model_cols + [self.ensemble_col]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, fmt=".3f")
        plt.title("Model Correlation Matrix - Malignancy")
        plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_report(self, output_dir: str) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("# Malignancy Model Failure Analysis Report\n")

        # Basic metrics
        metrics = self.analyze_basic_metrics()
        report.append("## 1. Basic Performance Metrics\n")
        for model, metric in metrics.items():
            report.append(f"### {model}")
            report.append(f"- AUROC: {metric['auroc']:.4f}")
            report.append(f"- Youden Threshold: {metric['youden_threshold']:.4f}")
            report.append(f"- Best F1 Threshold: {metric['best_f1_threshold']:.4f}")
            report.append(f"- Sensitivity at 95% Specificity: {metric['sensitivity_95_specificity']:.4f}")
            report.append(f"- Specificity at 95% Sensitivity: {metric['specificity_95_sensitivity']:.4f}\n")

        # Failure cases
        failures = self.identify_failure_cases()
        report.append("## 2. Failure Case Analysis\n")
        report.append(f"- False Positives (Benign predicted as Malignant): {len(failures['false_positives'])}")
        report.append(f"- False Negatives (Malignant predicted as Benign): {len(failures['false_negatives'])}")
        report.append(f"- High Confidence Errors: {len(failures['high_confidence_errors'])}")
        report.append(f"- Low Confidence Correct: {len(failures['low_confidence_correct'])}")
        report.append(f"- High Ensemble Disagreement: {len(failures['ensemble_disagreement'])}\n")

        # Ensemble methods comparison
        ensemble_methods = self.analyze_ensemble_methods()
        report.append("## 3. Ensemble Methods Comparison\n")
        for method, result in ensemble_methods.items():
            report.append(f"- {method.capitalize()}: AUROC = {result['auroc']:.4f}")
            if "weights" in result:
                report.append(f"  - Weights: {result['weights']}")
        report.append("")

        # Threshold optimization
        thresholds = self.optimize_thresholds()
        report.append("## 4. Optimal Thresholds\n")
        for threshold_name, threshold_value in thresholds.items():
            report.append(f"- {threshold_name}: {threshold_value:.4f}")

        # Save report
        report_path = f"{output_dir}/malignancy_failure_analysis_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))

        return report_path


@hydra.main(version_base="1.2", config_path="configs", config_name="config_failure")
def main(config: DictConfig):
    print_config(config, resolve=True)
    set_seed()

    # Load results from config
    result_path = config.result_csv_path
    logger.info(f"Loading results from: {result_path}")
    df = pd.read_csv(result_path)

    # For malignancy, use all data since there's no separate test set
    # All models were trained on all folds, so we evaluate on the entire dataset
    logger.info(f"Using all data for malignancy analysis: {len(df)} samples")

    # Extract model columns
    model_cols = [col for col in df.columns if col.startswith("prob_model_")]
    ensemble_col = "prob_ensemble"

    # Initialize analyzer
    analyzer = MalignancyFailureAnalyzer(df, model_cols, ensemble_col)

    # Create output directory
    output_dir = f"malignancy_failure_analysis_output/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate analysis
    logger.info("Generating malignancy failure analysis...")
    report_path = analyzer.generate_report(output_dir)

    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.create_visualizations(output_dir)

    # Print summary
    metrics = analyzer.analyze_basic_metrics()
    ensemble_methods = analyzer.analyze_ensemble_methods()

    logger.info("\n" + "=" * 50)
    logger.info("MALIGNANCY FAILURE ANALYSIS SUMMARY")
    logger.info("=" * 50)

    logger.info("\n📊 Model Performance:")
    for model, metric in metrics.items():
        logger.info(f"  {model}: AUROC = {metric['auroc']:.4f}")

    logger.info("\n🔧 Ensemble Methods:")
    for method, result in ensemble_methods.items():
        logger.info(f"  {method.capitalize()}: AUROC = {result['auroc']:.4f}")

    logger.info(f"\n📄 Detailed report saved to: {report_path}")
    logger.info(f"📊 Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
