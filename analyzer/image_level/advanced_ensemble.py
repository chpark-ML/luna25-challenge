import logging
import os
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from shared_lib.utils.utils import print_config, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedEnsembleAnalyzer:
    """Advanced ensemble analysis with multiple methods and optimization"""

    def __init__(self, df: pd.DataFrame, model_cols: List[str], ensemble_col: str = 'prob_ensemble', val_df: pd.DataFrame = None):
        self.df = df
        self.model_cols = model_cols
        self.ensemble_col = ensemble_col
        self.y_true = df['annotation']
        self.val_df = val_df

    def logistic_regression_ensemble(self) -> Dict:
        """Learn optimal weights using logistic regression"""
        if self.val_df is None:
            raise ValueError("Validation data is required for meta-learner training")

        # Prepare training data (val)
        X_train = self.val_df[self.model_cols].values
        y_train = self.val_df['annotation'].values

        # Prepare test data
        X_test = self.df[self.model_cols].values
        y_test = self.y_true.values

        # Fit logistic regression on val data
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)

        # Get predictions on test data
        lr_probs = lr.predict_proba(X_test)[:, 1]

        return {
            'probs': lr_probs,
            'weights': lr.coef_[0],
            'intercept': lr.intercept_[0],
            'auroc': roc_auc_score(y_test, lr_probs)
        }

    def random_forest_ensemble(self) -> Dict:
        """Use Random Forest as meta-learner"""
        if self.val_df is None:
            raise ValueError("Validation data is required for meta-learner training")

        # Prepare training data (val)
        X_train = self.val_df[self.model_cols].values
        y_train = self.val_df['annotation'].values

        # Prepare test data
        X_test = self.df[self.model_cols].values
        y_test = self.y_true.values

        # Fit Random Forest on val data
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Get predictions on test data
        rf_probs = rf.predict_proba(X_test)[:, 1]

        return {
            'probs': rf_probs,
            'feature_importance': rf.feature_importances_,
            'auroc': roc_auc_score(y_test, rf_probs)
        }

    def optimize_weights_gradient(self) -> Dict:
        """Optimize ensemble weights using gradient-based optimization"""
        if self.val_df is None:
            raise ValueError("Validation data is required for meta-learner training")

        # Use val data for optimization
        X_val = self.val_df[self.model_cols].values
        y_val = self.val_df['annotation'].values

        def objective(weights):
            # Normalize weights
            weights = np.abs(weights)  # Ensure positive weights
            weights = weights / np.sum(weights)

            # Calculate weighted ensemble on val data
            ensemble_probs = np.zeros(len(self.val_df))
            for i, col in enumerate(self.model_cols):
                ensemble_probs += weights[i] * self.val_df[col].values

            # Return negative AUROC (minimize negative AUROC = maximize AUROC)
            return -roc_auc_score(y_val, ensemble_probs)

        # Initial weights (equal)
        initial_weights = np.ones(len(self.model_cols)) / len(self.model_cols)

        # Optimize on val data
        result = minimize(objective, initial_weights, method='L-BFGS-B')

        # Get final weights and predictions on test data
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        ensemble_probs = np.zeros(len(self.df))
        for i, col in enumerate(self.model_cols):
            ensemble_probs += optimal_weights[i] * self.df[col].values

        return {
            'probs': ensemble_probs,
            'weights': optimal_weights,
            'auroc': roc_auc_score(self.y_true, ensemble_probs),
            'optimization_success': result.success
        }

    def bayesian_ensemble(self, alpha: float = 1.0, beta: float = 1.0) -> Dict:
        """Bayesian ensemble using Beta distribution"""
        # Convert probabilities to logits
        logits = np.log(self.df[self.model_cols] / (1 - self.df[self.model_cols] + 1e-10))

        # Bayesian averaging with Beta prior
        n_models = len(self.model_cols)
        posterior_alpha = alpha + np.sum(self.y_true)
        posterior_beta = beta + n_models - np.sum(self.y_true)

        # Calculate posterior mean
        bayesian_probs = posterior_alpha / (posterior_alpha + posterior_beta)

        # Weight by individual model performance
        individual_aurocs = [roc_auc_score(self.y_true, self.df[col]) for col in self.model_cols]
        weights = np.array(individual_aurocs) / sum(individual_aurocs)

        # Final ensemble
        ensemble_probs = np.zeros(len(self.df))
        for i, col in enumerate(self.model_cols):
            ensemble_probs += weights[i] * self.df[col].values

        # Combine with Bayesian prior
        final_probs = 0.7 * ensemble_probs + 0.3 * bayesian_probs

        return {
            'probs': final_probs,
            'weights': weights,
            'bayesian_prob': bayesian_probs,
            'auroc': roc_auc_score(self.y_true, final_probs)
        }

    def dynamic_ensemble(self, confidence_threshold: float = 0.1) -> Dict:
        """Dynamic ensemble based on model confidence"""
        # Calculate confidence for each model
        confidences = {}
        for col in self.model_cols:
            probs = self.df[col].values
            # Confidence is distance from 0.5
            confidences[col] = np.abs(probs - 0.5)

        # Dynamic ensemble
        dynamic_probs = np.zeros(len(self.df))
        total_weights = np.zeros(len(self.df))

        for col in self.model_cols:
            confidence = confidences[col]
            # Only use models with high confidence
            mask = confidence > confidence_threshold
            weights = confidence * mask
            dynamic_probs += weights * self.df[col].values
            total_weights += weights

        # Normalize
        dynamic_probs = np.where(total_weights > 0,
                               dynamic_probs / total_weights,
                               self.df[self.ensemble_col].values)

        return {
            'probs': dynamic_probs,
            'confidence_threshold': confidence_threshold,
            'auroc': roc_auc_score(self.y_true, dynamic_probs)
        }

    def optimize_threshold_comprehensive(self, method: str = 'f1') -> Dict:
        """Comprehensive threshold optimization for different metrics"""
        thresholds = {}

        for col in self.model_cols + [self.ensemble_col]:
            fpr, tpr, thresh = roc_curve(self.y_true, self.df[col])

            # Different optimization criteria
            if method == 'f1':
                # F1 score optimization
                best_score = 0
                best_threshold = 0.5
                for t in np.arange(0.1, 0.9, 0.01):
                    y_pred = (self.df[col] > t).astype(int)
                    score = f1_score(self.y_true, y_pred)
                    if score > best_score:
                        best_score = score
                        best_threshold = t
                thresholds[col] = best_threshold

            elif method == 'precision':
                # Precision optimization
                best_score = 0
                best_threshold = 0.5
                for t in np.arange(0.1, 0.9, 0.01):
                    y_pred = (self.df[col] > t).astype(int)
                    if np.sum(y_pred) > 0:  # Avoid division by zero
                        score = precision_score(self.y_true, y_pred)
                        if score > best_score:
                            best_score = score
                            best_threshold = t
                thresholds[col] = best_threshold

            elif method == 'recall':
                # Recall optimization
                best_score = 0
                best_threshold = 0.5
                for t in np.arange(0.1, 0.9, 0.01):
                    y_pred = (self.df[col] > t).astype(int)
                    score = recall_score(self.y_true, y_pred)
                    if score > best_score:
                        best_score = score
                        best_threshold = t
                thresholds[col] = best_threshold

            elif method == 'balanced_accuracy':
                # Balanced accuracy optimization
                best_score = 0
                best_threshold = 0.5
                for t in np.arange(0.1, 0.9, 0.01):
                    y_pred = (self.df[col] > t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    balanced_acc = (sensitivity + specificity) / 2
                    if balanced_acc > best_score:
                        best_score = balanced_acc
                        best_threshold = t
                thresholds[col] = best_threshold

        return thresholds

    def compare_all_methods(self) -> Dict:
        """Compare all ensemble methods"""
        methods = {}

        # Basic methods
        methods['average'] = {
            'probs': self.df[self.ensemble_col],
            'auroc': roc_auc_score(self.y_true, self.df[self.ensemble_col])
        }

        # Weighted by AUROC
        individual_aurocs = [roc_auc_score(self.y_true, self.df[col]) for col in self.model_cols]
        weights = np.array(individual_aurocs) / sum(individual_aurocs)
        weighted_probs = np.zeros(len(self.df))
        for i, col in enumerate(self.model_cols):
            weighted_probs += weights[i] * self.df[col].values
        methods['weighted_auroc'] = {
            'probs': weighted_probs,
            'auroc': roc_auc_score(self.y_true, weighted_probs),
            'weights': weights
        }

        # Advanced methods
        methods['logistic_regression'] = self.logistic_regression_ensemble()
        methods['random_forest'] = self.random_forest_ensemble()
        methods['gradient_optimization'] = self.optimize_weights_gradient()
        methods['bayesian'] = self.bayesian_ensemble()
        methods['dynamic'] = self.dynamic_ensemble()

        return methods

    def create_comprehensive_visualizations(self, output_dir: str):
        """Create comprehensive visualizations for all methods"""
        os.makedirs(output_dir, exist_ok=True)

        # Compare all methods
        methods = self.compare_all_methods()

        # 1. ROC curves for all methods
        plt.figure(figsize=(14, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        for i, (method_name, method_result) in enumerate(methods.items()):
            fpr, tpr, _ = roc_curve(self.y_true, method_result['probs'])
            auroc = method_result['auroc']
            plt.plot(fpr, tpr, label=f'{method_name} (AUROC = {auroc:.4f})',
                    color=colors[i], linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves - All Ensemble Methods', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_methods_roc.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. AUROC comparison bar plot
        plt.figure(figsize=(12, 8))
        method_names = list(methods.keys())
        aurocs = [methods[name]['auroc'] for name in method_names]

        bars = plt.bar(method_names, aurocs, color=colors[:len(method_names)])
        plt.xlabel('Ensemble Method', fontsize=14)
        plt.ylabel('AUROC', fontsize=14)
        plt.title('AUROC Comparison - All Ensemble Methods', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.8, 0.95)

        # Add value labels on bars
        for bar, auroc in zip(bars, aurocs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{auroc:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/auroc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Threshold optimization comparison
        threshold_methods = ['f1', 'precision', 'recall', 'balanced_accuracy']
        threshold_results = {}

        for method in threshold_methods:
            thresholds = self.optimize_threshold_comprehensive(method)
            threshold_results[method] = thresholds

        # Create threshold comparison plot
        plt.figure(figsize=(15, 8))
        models = list(threshold_results['f1'].keys())
        x = np.arange(len(models))
        width = 0.2

        for i, (thresh_method, thresh_dict) in enumerate(threshold_results.items()):
            values = [thresh_dict[model] for model in models]
            plt.bar(x + i*width, values, width, label=thresh_method, alpha=0.8)

        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Optimal Threshold', fontsize=14)
        plt.title('Optimal Thresholds by Optimization Method', fontsize=16)
        plt.xticks(x + width*1.5, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Model correlation and weights
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Correlation matrix
        corr_matrix = self.df[self.model_cols + [self.ensemble_col]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=axes[0,0])
        axes[0,0].set_title('Model Correlation Matrix')

        # Individual AUROC comparison
        individual_aurocs = [roc_auc_score(self.y_true, self.df[col]) for col in self.model_cols]
        axes[0,1].bar(self.model_cols, individual_aurocs, color='skyblue')
        axes[0,1].set_title('Individual Model AUROC')
        axes[0,1].set_ylabel('AUROC')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Weight comparison for different methods
        methods_with_weights = ['weighted_auroc', 'logistic_regression', 'gradient_optimization']
        for i, method in enumerate(methods_with_weights):
            if 'weights' in methods[method]:
                weights = methods[method]['weights']
                axes[1,0].bar([f'{method}_{j}' for j in range(len(weights))],
                             weights, alpha=0.7, label=method)
        axes[1,0].set_title('Model Weights by Method')
        axes[1,0].set_ylabel('Weight')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)

        # Probability distribution comparison
        for method_name, method_result in methods.items():
            if 'probs' in method_result:
                axes[1,1].hist(method_result['probs'], bins=50, alpha=0.5,
                              label=method_name, density=True)
        axes[1,1].set_title('Probability Distribution Comparison')
        axes[1,1].set_xlabel('Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_advanced_report(self, output_dir: str) -> str:
        """Generate comprehensive advanced analysis report"""
        report = []
        report.append("# Advanced Ensemble Analysis Report\n")

        # Compare all methods
        methods = self.compare_all_methods()

        report.append("## 1. Ensemble Methods Performance\n")
        report.append("| Method | AUROC | Description |")
        report.append("|--------|-------|-------------|")

        method_descriptions = {
            'average': 'Simple average of all models',
            'weighted_auroc': 'Weighted average based on individual AUROC',
            'logistic_regression': 'Logistic regression as meta-learner',
            'random_forest': 'Random Forest as meta-learner',
            'gradient_optimization': 'Gradient-based weight optimization',
            'bayesian': 'Bayesian ensemble with Beta prior',
            'dynamic': 'Dynamic ensemble based on confidence'
        }

        for method_name, method_result in methods.items():
            auroc = method_result['auroc']
            desc = method_descriptions.get(method_name, 'Advanced ensemble method')
            report.append(f"| {method_name} | {auroc:.4f} | {desc} |")

        report.append("")

        # Threshold optimization results
        report.append("## 2. Threshold Optimization Results\n")
        threshold_methods = ['f1', 'precision', 'recall', 'balanced_accuracy']

        for thresh_method in threshold_methods:
            thresholds = self.optimize_threshold_comprehensive(thresh_method)
            report.append(f"### {thresh_method.upper()} Optimization\n")
            for model, threshold in thresholds.items():
                report.append(f"- {model}: {threshold:.4f}")
            report.append("")

        # Detailed analysis of best method
        best_method = max(methods.items(), key=lambda x: x[1]['auroc'])
        report.append(f"## 3. Best Method Analysis: {best_method[0]}\n")
        report.append(f"- AUROC: {best_method[1]['auroc']:.4f}\n")

        if 'weights' in best_method[1]:
            report.append("### Model Weights:\n")
            for i, col in enumerate(self.model_cols):
                weight = best_method[1]['weights'][i]
                report.append(f"- {col}: {weight:.4f}\n")

        # Save report
        report_path = f"{output_dir}/advanced_ensemble_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        return report_path


@hydra.main(version_base="1.2", config_path="configs", config_name="config_inference")
def main(config: DictConfig):
    print_config(config, resolve=True)
    set_seed()

    # Load results from config
    result_path = config.result_csv_path
    logger.info(f"Loading results from: {result_path}")
    df = pd.read_csv(result_path)

    # Split into val and test data
    df_val = df[df["mode"] == "val"]
    df_test = df[df["mode"] == "test"]

    logger.info(f"Validation samples: {len(df_val)}")
    logger.info(f"Test samples: {len(df_test)}")

    # Extract model columns
    model_cols = [col for col in df.columns if col.startswith("prob_model_")]
    ensemble_col = "prob_ensemble"

    # Initialize analyzer with both val and test data
    analyzer = AdvancedEnsembleAnalyzer(df_test, model_cols, ensemble_col, df_val)

    # Create output directory
    output_dir = "advanced_ensemble_output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate analysis
    logger.info("Generating advanced ensemble analysis...")
    report_path = analyzer.generate_advanced_report(output_dir)

    # Create visualizations
    logger.info("Creating comprehensive visualizations...")
    analyzer.create_comprehensive_visualizations(output_dir)

    # Print summary
    methods = analyzer.compare_all_methods()

    print("\n" + "="*60)
    print("ADVANCED ENSEMBLE ANALYSIS SUMMARY")
    print("="*60)

    print("\n🏆 Method Performance Ranking:")
    sorted_methods = sorted(methods.items(), key=lambda x: x[1]['auroc'], reverse=True)
    for i, (method_name, method_result) in enumerate(sorted_methods, 1):
        print(f"  {i}. {method_name}: AUROC = {method_result['auroc']:.4f}")

    best_method = sorted_methods[0]
    print(f"\n🥇 Best Method: {best_method[0]} (AUROC = {best_method[1]['auroc']:.4f})")

    if 'weights' in best_method[1]:
        print("\n📊 Model Weights:")
        for i, col in enumerate(model_cols):
            weight = best_method[1]['weights'][i]
            print(f"  {col}: {weight:.4f}")

    print(f"\n📄 Detailed report saved to: {report_path}")
    print(f"📊 Visualizations saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    main()
