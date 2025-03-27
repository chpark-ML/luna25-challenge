"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from helpers import run_prediction_processing, tree
from sklearn.metrics import roc_auc_score, roc_curve

# INPUT_DIRECTORY = Path("./test/input")
# OUTPUT_DIRECTORY = Path("./test/output")
# GROUND_TRUTH_DIRECTORY = Path("ground_truth")

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")


def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_lung_nodule_malignancy_likelihoods = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="nodule-locations-likelihoods",
    )

    # Secondly, read the results
    result_lung_nodule_malignancy_risk = load_json_file(
        location=location_lung_nodule_malignancy_likelihoods,
    )
    annotation_ids = [case["name"] for case in result_lung_nodule_malignancy_risk["points"]]
    malignancy_risks = [case["probability"] for case in result_lung_nodule_malignancy_risk["points"]]

    # # Thirdly, retrieve the input file name to match it with your ground truth
    # image_name_chest_ct = get_image_name(
    #     values=job["inputs"],
    #     slug="chest-ct",
    # )

    # Fourthly, load your ground truth for this image
    ground_truths = load_ground_truth_file(path_to_ground_truth=GROUND_TRUTH_DIRECTORY)

    # process the results
    results = []

    for index, (annotation_id, malignancy_risk) in enumerate(zip(annotation_ids, malignancy_risks)):
        ground_truth = ground_truths[ground_truths["AnnotationID"] == annotation_id]["label"].values[0]
        results.append({"AnnotationID": annotation_id, "label": ground_truth, "risk_prediction": malignancy_risk})

    return results


def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    results = run_prediction_processing(fn=process, predictions=predictions)

    results = pd.DataFrame([item for sublist in results for item in sublist])

    auc_testing_cohort = calculate_auc(results["label"], results["risk_prediction"])
    sens_testing_cohort = calculate_sensitivity(results["label"], results["risk_prediction"])
    spec_testing_cohort = calculate_specificity(results["label"], results["risk_prediction"])

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission

    metrics["Final Score"] = float(auc_testing_cohort["auc"])
    metrics["AUC"] = float(auc_testing_cohort["auc"])
    metrics["AUC 95% CI lower bound"] = float(auc_testing_cohort["ci_lower"])
    metrics["AUC 95% CI upper bound"] = float(auc_testing_cohort["ci_upper"])
    metrics["Sensitivity"] = float(sens_testing_cohort["sensitivity"])
    metrics["Specificity"] = float(spec_testing_cohort["specificity"])

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    print("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        print(line)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_ground_truth_file(path_to_ground_truth):
    # Load the ground truth file
    ground_truths = pd.read_csv(path_to_ground_truth / "luna25_hidden_validation.csv")

    return ground_truths


def calculate_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)

    # Bootstrapping for 95% confidence intervals
    n_bootstraps = 1000
    rng = np.random.RandomState(seed=42)
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        # Resample the data
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true.iloc[indices])) < 2:
            # Skip this resample if only one class is present
            continue
        score = roc_auc_score(y_true.iloc[indices], y_pred.iloc[indices])
        bootstrapped_aucs.append(score)

    # Calculate the confidence intervals
    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    return {"auc": auc, "ci_lower": ci_lower, "ci_upper": ci_upper}


def calculate_sensitivity(y_true, y_pred):
    """
    Computes the sensitivity (recall) at 95% specificity for a classifier.

    Parameters:
        y_true (array-like): Ground truth binary labels (0 = benign, 1 = malignant).
        y_pred (array-like): Predicted probability scores from the classifier.

    Returns:
        float: Sensitivity (recall) at 95% specificity.
        float: Decision threshold used to achieve 95% specificity.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find the threshold corresponding to 95% specificity (FPR = 1 - specificity)
    target_fpr = 1 - 0.95  # 5% false positive rate
    idx = np.where(fpr <= target_fpr)[0][-1]  # Get the last index where FPR <= 5%

    # Extract sensitivity (TPR) and threshold
    sensitivity = tpr[idx]

    return {"sensitivity": sensitivity}


def calculate_specificity(y_true, y_pred):
    """
    Computes the specificity at 95% sensitivity for a classifier.

    Parameters:
        y_true (array-like): Ground truth binary labels (0 = benign, 1 = malignant).
        y_pred (array-like): Predicted probability scores from the classifier.

    Returns:
        float: Specificity at 95% sensitivity.
        float: Decision threshold used to achieve 95% sensitivity.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find the threshold corresponding to 95% sensitivity (TPR = 0.95)
    target_tpr = 0.95  # Sensitivity (TPR) threshold
    idx = np.where(tpr >= target_tpr)[0][0]  # Get first index where TPR >= 95%

    # Extract specificity (1 - FPR) and threshold
    specificity = 1 - fpr[idx]

    return {"specificity": specificity}


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
