import numpy as np
import pandas as pd


def split_calibration_evaluation(
    df,
    calibration_size=60,
    random_seed=42
):
    """
    Split synthetic sensor data into:

    1. Calibration set:
       Known-normal observations used to estimate alert thresholds.

    2. Evaluation set:
       Remaining normal observations + all anomaly observations.
    """

    normal_df = df[
        df["is_anomaly"] == 0
    ].copy()

    anomaly_df = df[
        df["is_anomaly"] == 1
    ].copy()

    calibration_df = normal_df.sample(
        n=calibration_size,
        random_state=random_seed
    )

    remaining_normal_df = normal_df.drop(
        calibration_df.index
    )

    evaluation_df = pd.concat(
        [
            remaining_normal_df,
            anomaly_df
        ],
        ignore_index=True
    )

    evaluation_df = evaluation_df.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    calibration_df = (
        calibration_df
        .reset_index(drop=True)
    )

    return calibration_df, evaluation_df


def mean_std_threshold(
    calibration_df,
    metric_cols,
    k=2.5
):
    """
    Upper alert threshold:
        mean + k * standard deviation
    """

    thresholds = {}

    for metric in metric_cols:

        values = calibration_df[
            metric
        ].to_numpy()

        threshold = (
            np.mean(values)
            + k * np.std(
                values,
                ddof=1
            )
        )

        thresholds[metric] = float(
            threshold
        )

    return thresholds


if __name__ == "__main__":

    from data_generator import (
        generate_sensor_data,
        METRIC_NAMES
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

    print(
        "Calibration shape:",
        calibration_df.shape
    )

    print(
        "\nCalibration class counts:"
    )
    print(
        calibration_df[
            "is_anomaly"
        ].value_counts()
    )

    print(
        "\nEvaluation shape:",
        evaluation_df.shape
    )

    print(
        "\nEvaluation class counts:"
    )
    print(
        evaluation_df[
            "is_anomaly"
        ].value_counts()
    )

    print(
        "\nMean + 2.5 SD thresholds:"
    )

    for metric, threshold in thresholds.items():
        print(
            f"{metric}: "
            f"{threshold:.3f}"
        )

def median_std_threshold(
    calibration_df,
    metric_cols,
    k=2.5
):
    """
    Upper alert threshold:
        median + k * standard deviation
    """

    thresholds = {}

    for metric in metric_cols:

        values = calibration_df[
            metric
        ].to_numpy()

        threshold = (
            np.median(values)
            + k * np.std(
                values,
                ddof=1
            )
        )

        thresholds[metric] = float(
            threshold
        )

    return thresholds


def quantile_threshold(
    calibration_df,
    metric_cols,
    q=0.99
):
    """
    Upper alert threshold based on
    an empirical quantile.
    """

    thresholds = {}

    for metric in metric_cols:

        values = calibration_df[
            metric
        ].to_numpy()

        threshold = np.quantile(
            values,
            q
        )

        thresholds[metric] = float(
            threshold
        )

    return thresholds


def median_mad_threshold(
    calibration_df,
    metric_cols,
    k=3.0
):
    """
    Robust upper alert threshold:
        median + k * scaled MAD

    Scaled MAD approximates standard
    deviation under normality.
    """

    thresholds = {}

    for metric in metric_cols:

        values = calibration_df[
            metric
        ].to_numpy()

        median = np.median(values)

        mad = np.median(
            np.abs(
                values - median
            )
        )

        scaled_mad = 1.4826 * mad

        threshold = (
            median
            + k * scaled_mad
        )

        thresholds[metric] = float(
            threshold
        )

    return thresholds

def get_all_threshold_methods(
    calibration_df,
    metric_cols
):
    """
    Generate thresholds using
    multiple estimation methods.
    """

    return {
        "mean_std":
            mean_std_threshold(
                calibration_df,
                metric_cols,
                k=2.5
            ),

        "median_std":
            median_std_threshold(
                calibration_df,
                metric_cols,
                k=2.5
            ),

        "quantile_99":
            quantile_threshold(
                calibration_df,
                metric_cols,
                q=0.99
            ),

        "median_mad":
            median_mad_threshold(
                calibration_df,
                metric_cols,
                k=3.0
            )
    }