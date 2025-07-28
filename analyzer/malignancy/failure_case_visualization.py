import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
import seaborn as sns
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data_lake.constants import DB_ADDRESS, TARGET_COLLECTION, TARGET_DB, DBKey
from shared_lib.tools.image_parser import extract_patch
from shared_lib.utils.utils import print_config, set_seed
from shared_lib.utils.utils_vis import save_plot
from trainer.downstream.datasets.luna25 import extract_patch_dicom_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CLIENT = pymongo.MongoClient(DB_ADDRESS)


class FailureCaseVisualizer:
    """Visualize failure cases for malignancy models"""

    def __init__(self, df: pd.DataFrame, model_cols: List[str], ensemble_col: str = "prob_ensemble"):
        self.df = df
        self.model_cols = model_cols
        self.ensemble_col = ensemble_col
        self.y_true = df["annotation"].astype(float)

        # Calculate ensemble statistics
        self._calculate_ensemble_stats()

    def _calculate_ensemble_stats(self):
        """Calculate ensemble variance, entropy, and other statistics"""
        # Ensemble variance
        self.df["ensemble_variance"] = self.df[self.model_cols].var(axis=1)

        # Binary entropy for each model
        binary_entropies = []
        for _, row in self.df[self.model_cols].iterrows():
            entropies = -(row.values * np.log2(row.values + 1e-10) + (1 - row.values) * np.log2(1 - row.values + 1e-10))
            binary_entropies.append(np.mean(entropies))
        self.df["prob_entropy"] = binary_entropies

        # Model agreement (standard deviation)
        self.df["model_std"] = self.df[self.model_cols].std(axis=1)

        # Confidence score (distance from 0.5)
        self.df["confidence"] = np.abs(self.df[self.ensemble_col] - 0.5)

    def identify_failure_cases(self, threshold: float = 0.5, top_k: int = 20) -> Dict[str, pd.DataFrame]:
        """Identify different types of failure cases"""
        failure_cases = {}

        # 1. False Positives (Benign predicted as Malignant)
        fp_mask = (self.y_true == 0.0) & (self.df[self.ensemble_col] > threshold)
        fp_df = self.df[fp_mask].copy()
        fp_df = fp_df.sort_values(by=self.ensemble_col, ascending=False).head(top_k)
        fp_df["failure_type"] = "False Positive"
        failure_cases["false_positives"] = fp_df

        # 2. False Negatives (Malignant predicted as Benign)
        fn_mask = (self.y_true == 1.0) & (self.df[self.ensemble_col] < threshold)
        fn_df = self.df[fn_mask].copy()
        fn_df = fn_df.sort_values(by=self.ensemble_col, ascending=True).head(top_k)
        fn_df["failure_type"] = "False Negative"
        failure_cases["false_negatives"] = fn_df

        # 3. High Confidence Errors (very confident but wrong)
        high_conf_error_mask = ((self.y_true == 0.0) & (self.df[self.ensemble_col] > 0.8)) | (
            (self.y_true == 1.0) & (self.df[self.ensemble_col] < 0.2)
        )
        hce_df = self.df[high_conf_error_mask].copy()
        hce_df = hce_df.sort_values(by="confidence", ascending=False).head(top_k)
        hce_df["failure_type"] = "High Confidence Error"
        failure_cases["high_confidence_errors"] = hce_df

        # 4. High Ensemble Disagreement (models disagree a lot)
        high_disagreement_mask = self.df["ensemble_variance"] > self.df["ensemble_variance"].quantile(0.95)
        hd_df = self.df[high_disagreement_mask].copy()
        hd_df = hd_df.sort_values(by="ensemble_variance", ascending=False).head(top_k)
        hd_df["failure_type"] = "High Disagreement"
        failure_cases["high_disagreement"] = hd_df

        # 5. Low Confidence Correct (uncertain but correct)
        low_conf_correct_mask = (
            (self.y_true == 0.0) & (self.df[self.ensemble_col] < threshold) & (self.df[self.ensemble_col] > 0.3)
        ) | ((self.y_true == 1.0) & (self.df[self.ensemble_col] > threshold) & (self.df[self.ensemble_col] < 0.7))
        lcc_df = self.df[low_conf_correct_mask].copy()
        lcc_df = lcc_df.sort_values(by="confidence", ascending=True).head(top_k)
        lcc_df["failure_type"] = "Low Confidence Correct"
        failure_cases["low_confidence_correct"] = lcc_df

        # 6. Edge Cases (probability very close to threshold)
        edge_mask = np.abs(self.df[self.ensemble_col] - threshold) < 0.05
        edge_df = self.df[edge_mask].copy()
        edge_df = edge_df.sort_values(by="ensemble_variance", ascending=False).head(top_k)
        edge_df["failure_type"] = "Edge Case"
        failure_cases["edge_cases"] = edge_df

        return failure_cases

    def create_summary_visualization(self, failure_cases: Dict[str, pd.DataFrame], output_dir: str):
        """Create summary visualizations for failure cases"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Failure case counts
        plt.figure(figsize=(12, 6))
        case_counts = {case_type: len(df) for case_type, df in failure_cases.items()}
        plt.bar(case_counts.keys(), case_counts.values(), color="lightcoral")
        plt.title("Number of Failure Cases by Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/failure_case_counts.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Probability distribution for each failure type
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (case_type, df) in enumerate(failure_cases.items()):
            if i < len(axes) and len(df) > 0:
                axes[i].hist(df[self.ensemble_col], bins=20, alpha=0.7, color="lightcoral")
                axes[i].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
                axes[i].set_title(f'{case_type.replace("_", " ").title()}\n(n={len(df)})')
                axes[i].set_xlabel("Ensemble Probability")
                axes[i].set_ylabel("Count")
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/failure_case_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Ensemble variance vs probability for failure cases
        plt.figure(figsize=(14, 10))
        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        for i, (case_type, df) in enumerate(failure_cases.items()):
            if len(df) > 0:
                plt.scatter(
                    df["ensemble_variance"],
                    df[self.ensemble_col],
                    c=colors[i % len(colors)],
                    alpha=0.6,
                    label=f'{case_type.replace("_", " ").title()} (n={len(df)})',
                    s=50,
                )

        plt.xlabel("Ensemble Variance")
        plt.ylabel("Ensemble Probability")
        plt.title("Ensemble Variance vs Probability for Failure Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/variance_vs_probability.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Model agreement heatmap for each failure type
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (case_type, df) in enumerate(failure_cases.items()):
            if i < len(axes) and len(df) > 0:
                # Calculate correlation matrix for this failure type
                corr_matrix = df[self.model_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, fmt=".2f", ax=axes[i])
                axes[i].set_title(f'{case_type.replace("_", " ").title()}\nModel Correlations')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_correlations_by_failure_type.png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_failure_case_images(self, failure_cases: Dict[str, pd.DataFrame], output_dir: str):
        """Save 3D visualizations for failure cases"""
        logger.info("Starting failure case image visualization...")

        # Combine all failure cases
        all_failure_df = pd.concat(failure_cases.values(), ignore_index=True)

        # Get annotation IDs for MongoDB query
        annotation_ids = all_failure_df["annot_ids"].tolist()
        query = {"annotation_id": {"$in": annotation_ids}}
        projection = {}

        # Fetch documents from MongoDB
        logger.info(f"Fetching {len(annotation_ids)} documents from MongoDB...")
        nodule_candidates = list(_CLIENT[TARGET_DB][TARGET_COLLECTION].find(query, projection))

        # Convert to DataFrame and merge
        db_df = pd.DataFrame(nodule_candidates)
        merged_df = all_failure_df.merge(db_df, left_on="annot_ids", right_on="annotation_id", how="left")

        # Save images for each failure type
        for failure_type, failure_df in failure_cases.items():
            if len(failure_df) == 0:
                continue

            failure_output_dir = os.path.join(output_dir, failure_type)
            os.makedirs(failure_output_dir, exist_ok=True)

            # Get merged data for this failure type
            failure_merged = merged_df[merged_df["failure_type"] == failure_df["failure_type"].iloc[0]]

            logger.info(f"Saving {len(failure_merged)} images for {failure_type}...")
            self._save_failure_images(failure_merged, failure_output_dir, failure_type)

    def _save_failure_images(self, df: pd.DataFrame, output_dir: str, failure_type: str):
        """Save individual failure case images"""
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {failure_type}"):
            try:
                annotation = int(row["annotation"])
                annotation_id = row["annotation_id"]
                prob_ensemble = row["prob_ensemble"]
                ensemble_variance = row.get("ensemble_variance", 0)

                # Create filename with detailed info
                filename = f"{annotation_id}_gt{annotation}_pred{prob_ensemble:.3f}_var{ensemble_variance:.4f}.png"
                save_path = Path(os.path.join(output_dir, filename))

                # Skip if already exists
                if save_path.exists():
                    continue

                # Extract image data
                h5_path = row[DBKey.H5_PATH_NFS]
                d_coord_zyx = row[DBKey.D_COORD_ZYX]
                origin = row[DBKey.ORIGIN]
                transform = row[DBKey.TRANSFORM]
                spacing = row[DBKey.SPACING]

                # Image extraction parameters
                size_xy = 128
                size_z = 64
                size_mm = 50
                size_px_xy = 72
                size_px_z = 48

                patch_size = [size_z, size_xy, size_xy]
                output_shape = (size_px_z, size_px_xy, size_px_xy)

                # Extract patch
                img = extract_patch_dicom_space(h5_path, d_coord_zyx, xy_size=size_xy, z_size=size_z)

                patch = extract_patch(
                    CTData=img,
                    coord=tuple(np.array(patch_size) // 2),
                    srcVoxelOrigin=np.array(origin),
                    srcWorldMatrix=np.array(transform),
                    srcVoxelSpacing=np.array(spacing),
                    output_shape=output_shape,
                    voxel_spacing=(
                        size_mm / size_px_z,
                        size_mm / size_px_xy,
                        size_mm / size_px_xy,
                    ),
                    rotations=None,
                    translations=None,
                    coord_space_world=False,
                    mode="3D",
                    order=1,
                )
                patch = np.squeeze(patch)

                # Prepare metadata
                attr = {
                    "annotation_id": annotation_id,
                    "ground_truth": annotation,
                    "prob_ensemble": prob_ensemble,
                    "ensemble_variance": ensemble_variance,
                    "failure_type": failure_type,
                }

                # Add individual model predictions
                for col in self.model_cols:
                    if col in row:
                        attr[col] = row[col]

                # Create figure title
                figure_title = f"{failure_type}: GT={annotation}, Pred={prob_ensemble:.3f}, Var={ensemble_variance:.4f}"

                # Save visualization
                save_plot(
                    input_image=patch,
                    mask_image=None,
                    nodule_zyx=None,
                    figure_title=figure_title,
                    meta=attr,
                    use_norm=True,
                    save_dir=str(save_path),
                    dpi=60,
                )

            except Exception as e:
                logger.warning(f"Failed to process {row.get('annotation_id', 'unknown')}: {str(e)}")
                continue

    def generate_failure_report(self, failure_cases: Dict[str, pd.DataFrame], output_dir: str) -> str:
        """Generate detailed failure analysis report"""
        report = []
        report.append("# Malignancy Model Failure Case Analysis Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary statistics
        total_samples = len(self.df)
        total_failures = sum(len(df) for df in failure_cases.values())

        report.append("## Summary Statistics\n")
        report.append(f"- Total samples analyzed: {total_samples}")
        report.append(f"- Total failure cases identified: {total_failures}")
        report.append(f"- Failure rate: {total_failures/total_samples*100:.2f}%\n")

        # Detailed breakdown
        report.append("## Failure Case Breakdown\n")
        for case_type, df in failure_cases.items():
            if len(df) > 0:
                report.append(f"### {case_type.replace('_', ' ').title()}")
                report.append(f"- Count: {len(df)}")
                report.append(f"- Mean ensemble probability: {df[self.ensemble_col].mean():.4f}")
                report.append(f"- Mean ensemble variance: {df['ensemble_variance'].mean():.6f}")
                report.append(f"- Mean confidence: {df['confidence'].mean():.4f}")

                # Show top 5 cases
                report.append(f"- Top 5 cases by ensemble probability:")
                top_cases = df.nlargest(5, self.ensemble_col)
                for _, case in top_cases.iterrows():
                    report.append(
                        f"  - ID: {case['annot_ids']}, GT: {int(case['annotation'])}, "
                        f"Pred: {case[self.ensemble_col]:.4f}, Var: {case['ensemble_variance']:.6f}"
                    )
                report.append("")

        # Model performance on failure cases
        report.append("## Model Performance on Failure Cases\n")
        for case_type, df in failure_cases.items():
            if len(df) > 0:
                report.append(f"### {case_type.replace('_', ' ').title()}")
                for col in self.model_cols:
                    if col in df.columns:
                        auroc = roc_auc_score(df["annotation"], df[col]) if len(df["annotation"].unique()) > 1 else 0.5
                        report.append(f"- {col}: AUROC = {auroc:.4f}, Mean Prob = {df[col].mean():.4f}")
                report.append("")

        # Recommendations
        report.append("## Recommendations\n")
        report.append("### False Positives")
        report.append("- Consider adding more diverse benign cases to training data")
        report.append("- Review feature extraction for cases with high false positive rates")
        report.append("- Consider ensemble methods that are more conservative\n")

        report.append("### False Negatives")
        report.append("- Review malignant cases that are consistently missed")
        report.append("- Consider data augmentation for underrepresented malignant patterns")
        report.append("- Analyze if missed cases have specific characteristics\n")

        report.append("### High Disagreement Cases")
        report.append("- Review these cases for potential labeling errors")
        report.append("- Consider using disagreement as an uncertainty measure")
        report.append("- May indicate need for expert review in clinical setting\n")

        # Save report
        report_path = os.path.join(output_dir, "failure_case_analysis_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report))

        return report_path


@hydra.main(version_base="1.2", config_path="configs", config_name="config_failure")
def main(config: DictConfig):
    print_config(config, resolve=True)
    set_seed()

    # Load results
    result_csv_path = config.result_csv_path
    logger.info(f"Loading results from: {result_csv_path}")
    df = pd.read_csv(result_csv_path)

    # Use test data for analysis
    test_df = df[df["mode"] == "test"].copy()
    logger.info(f"Using test data: {len(test_df)} samples")

    # Extract model columns
    model_cols = [col for col in test_df.columns if col.startswith("prob_model_")]
    ensemble_col = "prob_ensemble"

    logger.info(f"Found {len(model_cols)} individual models")

    # Initialize visualizer
    visualizer = FailureCaseVisualizer(test_df, model_cols, ensemble_col)

    # Identify failure cases
    logger.info("Identifying failure cases...")
    failure_cases = visualizer.identify_failure_cases(threshold=0.5, top_k=20)

    # Create output directory
    output_dir = f"failure_case_visualization_output/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FAILURE CASE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for case_type, df in failure_cases.items():
        logger.info(f"{case_type.replace('_', ' ').title()}: {len(df)} cases")
    # Generate summary visualizations
    logger.info("Creating summary visualizations...")
    visualizer.create_summary_visualization(failure_cases, output_dir)

    # Generate detailed report
    logger.info("Generating failure analysis report...")
    report_path = visualizer.generate_failure_report(failure_cases, output_dir)

    # Save failure case images
    logger.info("Saving failure case images...")
    visualizer.save_failure_case_images(failure_cases, output_dir)

    logger.info(f"\n📊 Summary visualizations saved to: {output_dir}/")
    logger.info(f"📄 Detailed report saved to: {report_path}")
    logger.info(f"🖼️  Failure case images saved to: {output_dir}/[failure_type]/")


if __name__ == "__main__":
    main()
