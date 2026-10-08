import pandas as pd

from data_generator import (
    generate_sensor_data,
    METRIC_NAMES
)

from threshold_methods import (
    split_calibration_evaluation,
    get_all_threshold_methods
)

from evaluation import (
    apply_thresholds,
    calculate_classification_metrics
)


def compare_threshold_methods():

    df = generate_sensor_data()

    calibration_df, evaluation_df = (
        split_calibration_evaluation(
            df,
            calibration_size=60,
            random_seed=42
        )
    )

    threshold_methods = (
        get_all_threshold_methods(
            calibration_df,
            METRIC_NAMES
        )
    )

    rows = []

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

        row = {
            "method": method_name,

            "temperature_threshold":
                thresholds["temperature"],

            "vibration_threshold":
                thresholds["vibration"],

            "power_threshold":
                thresholds["power"],

            **metrics
        }

        rows.append(row)

    comparison_df = pd.DataFrame(
        rows
    )

    return comparison_df


if __name__ == "__main__":

    comparison_df = (
        compare_threshold_methods()
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    print(
        comparison_df.round(4)
    )

    comparison_df.to_csv(
        "data/threshold_method_comparison.csv",
        index=False
    )