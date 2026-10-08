import numpy as np
import pandas as pd

from data_generator import (
    generate_sensor_data,
    METRIC_NAMES
)

from threshold_methods import (
    get_all_threshold_methods
)

from evaluation import (
    apply_thresholds,
    calculate_classification_metrics
)


def run_monte_carlo_stability(
    n_iterations=1000,
    calibration_size=60,
    random_seed=42
):
    """
    Repeatedly sample calibration datasets from the normal population,
    estimate thresholds, and evaluate downstream anomaly detection risk.
    """

    df = generate_sensor_data(
        random_seed=random_seed
    )

    normal_df = df[
        df["is_anomaly"] == 0
    ].copy()

    anomaly_df = df[
        df["is_anomaly"] == 1
    ].copy()

    rng = np.random.default_rng(
        random_seed
    )

    rows = []

    for iteration in range(n_iterations):

        # ---------------------------------
        # 1. Sample calibration observations
        # ---------------------------------

        calibration_indices = rng.choice(
            normal_df.index.to_numpy(),
            size=calibration_size,
            replace=False
        )

        calibration_df = normal_df.loc[
            calibration_indices
        ].copy()

        remaining_normal_df = normal_df.drop(
            calibration_indices
        )

        evaluation_df = pd.concat(
            [
                remaining_normal_df,
                anomaly_df
            ],
            ignore_index=True
        )

        # ---------------------------------
        # 2. Estimate thresholds
        # ---------------------------------

        threshold_methods = (
            get_all_threshold_methods(
                calibration_df,
                METRIC_NAMES
            )
        )

        # ---------------------------------
        # 3. Evaluate each method
        # ---------------------------------

        for method_name, thresholds \
                in threshold_methods.items():

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

            rows.append({
                "iteration": iteration,
                "method": method_name,

                "temperature_threshold":
                    thresholds["temperature"],

                "vibration_threshold":
                    thresholds["vibration"],

                "power_threshold":
                    thresholds["power"],

                "false_positive_rate":
                    metrics[
                        "false_positive_rate"
                    ],

                "false_negative_rate":
                    metrics[
                        "false_negative_rate"
                    ],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"]
            })

    return pd.DataFrame(rows)


def summarize_monte_carlo(
    results_df
):
    """
    Summarize threshold stability and downstream risk variability.
    """

    summary = (
        results_df
        .groupby("method")
        .agg(
            temperature_threshold_mean=(
                "temperature_threshold",
                "mean"
            ),

            temperature_threshold_std=(
                "temperature_threshold",
                "std"
            ),

            vibration_threshold_mean=(
                "vibration_threshold",
                "mean"
            ),

            vibration_threshold_std=(
                "vibration_threshold",
                "std"
            ),

            power_threshold_mean=(
                "power_threshold",
                "mean"
            ),

            power_threshold_std=(
                "power_threshold",
                "std"
            ),

            false_positive_rate_mean=(
                "false_positive_rate",
                "mean"
            ),

            false_positive_rate_std=(
                "false_positive_rate",
                "std"
            ),

            false_negative_rate_mean=(
                "false_negative_rate",
                "mean"
            ),

            false_negative_rate_std=(
                "false_negative_rate",
                "std"
            ),

            precision_mean=(
                "precision",
                "mean"
            ),

            recall_mean=(
                "recall",
                "mean"
            )
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":

    results_df = (
        run_monte_carlo_stability(
            n_iterations=1000,
            calibration_size=60,
            random_seed=42
        )
    )

    summary_df = (
        summarize_monte_carlo(
            results_df
        )
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    print(
        summary_df.round(4)
    )

    results_df.to_csv(
        "data/monte_carlo_threshold_results.csv",
        index=False
    )

    summary_df.to_csv(
        "data/monte_carlo_threshold_summary.csv",
        index=False
    )