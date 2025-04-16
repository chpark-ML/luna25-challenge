from trainer.common.utils.utils_mlflow import get_mlflow_cv_summary, save_table_as_image, set_mlflow_tracking_uri

# Mapping of MLflow metric names to readable names
METRIC_NAME_MAPPER = {
    "checkpoint_test-acc_c_malignancy_logistic": "Accuracy",
    "checkpoint_test-auroc_c_malignancy_logistic": "AUROC",
    "checkpoint_test-f1_c_malignancy_logistic": "F1 Score",
}


if __name__ == "__main__":
    # Set MLflow tracking server URL (Update with your actual MLflow server URL)
    mlflow_url = "http://172.31.10.111:18002"
    set_mlflow_tracking_uri(mlflow_url)

    # List of run names (latest versions will be selected if duplicates exist)
    experiment_name = "lct-malignancy-attr"
    num_fold = 6
    run_names = [f"cls_fine_model_0_val_fold{idx}_c_malignancy_logistic" for idx in range(num_fold)]

    try:
        # Generate the cross-validation summary
        summary_df = get_mlflow_cv_summary(experiment_name, run_names, metric_name_mapper=METRIC_NAME_MAPPER)

        # Display results in the console
        print("\nCross-Validation Performance Summary (Aggregated Across Runs):")
        print(summary_df.to_markdown())

        # Save the table as an image
        save_table_as_image(summary_df)

    except ValueError as e:
        print(f"Error: {e}")
