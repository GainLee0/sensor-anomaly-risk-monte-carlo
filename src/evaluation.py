import pandas as pd


def apply_thresholds(
    df,
    thresholds,
    metric_cols
):
    """
    Classify each observation using upper alert thresholds.

    Alert = 1 if ANY metric exceeds its threshold.
    Alert = 0 otherwise.
    """

    result_df = df.copy()

    alert_mask = False

    for metric in metric_cols:
        alert_mask = (
            alert_mask
            | (
                result_df[metric]
                > thresholds[metric]
            )
        )

    result_df["predicted_anomaly"] = (
        alert_mask.astype(int)
    )

    return result_df


def calculate_classification_metrics(
    result_df
):
    """
    Calculate basic anomaly detection metrics.
    """

    actual = result_df["is_anomaly"]
    predicted = result_df["predicted_anomaly"]

    true_positive = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    false_positive = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    true_negative = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

    false_negative = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
    )

    false_negative_rate = (
        false_negative
        / (false_negative + true_positive)
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    return {
        "TP": true_positive,
        "FP": false_positive,
        "TN": true_negative,
        "FN": false_negative,
        "false_positive_rate":
            false_positive_rate,
        "false_negative_rate":
            false_negative_rate,
        "precision":
            precision,
        "recall":
            recall
    }


if __name__ == "__main__":

    from data_generator import (
        generate_sensor_data,
        METRIC_NAMES
    )

    from threshold_methods import (
        split_calibration_evaluation,
        mean_std_threshold
    )

    df = generate_sensor_data()

    calibration_df, evaluation_df = (
        split_calibration_evaluation(
            df,
            calibration_size=60,
            random_seed=42
        )
    )

    thresholds = mean_std_threshold(
        calibration_df,
        METRIC_NAMES,
        k=2.5
    )

    results = apply_thresholds(
        evaluation_df,
        thresholds,
        METRIC_NAMES
    )

    metrics = (
        calculate_classification_metrics(
            results
        )
    )

    print("Classification results:")

    for key, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )