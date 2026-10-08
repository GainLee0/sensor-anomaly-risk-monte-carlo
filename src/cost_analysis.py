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


FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0


def calculate_expected_cost(
    false_positive,
    false_negative,
    false_positive_cost=FALSE_POSITIVE_COST,
    false_negative_cost=FALSE_NEGATIVE_COST
):
    """
    Calculate total decision cost.

    Default assumption:
        False Positive cost = 1
        False Negative cost = 5
    """

    return (
        false_positive * false_positive_cost
        + false_negative * false_negative_cost
    )


def compare_method_costs():

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

    for method_name, thresholds in threshold_methods.items():

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

        total_cost = calculate_expected_cost(
            false_positive=metrics["FP"],
            false_negative=metrics["FN"]
        )

        rows.append({
            "method": method_name,
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "false_positive_rate":
                metrics["false_positive_rate"],
            "false_negative_rate":
                metrics["false_negative_rate"],
            "expected_cost":
                total_cost
        })

    cost_df = pd.DataFrame(rows)

    return cost_df.sort_values(
        "expected_cost",
        ascending=True
    ).reset_index(drop=True)


if __name__ == "__main__":

    cost_df = compare_method_costs()

    print(cost_df.round(4))

    cost_df.to_csv(
        "data/threshold_cost_comparison.csv",
        index=False
    )