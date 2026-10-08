import pandas as pd
import matplotlib.pyplot as plt

from cost_analysis import (
    compare_method_costs,
    calculate_expected_cost
)


def run_cost_sensitivity(
    fn_cost_values=(1, 2, 5, 10, 20),
    fp_cost=1.0
):
    """
    Recalculate expected cost across different
    false-negative cost assumptions.
    """

    base_df = compare_method_costs()

    rows = []

    for fn_cost in fn_cost_values:

        for _, row in base_df.iterrows():

            cost = calculate_expected_cost(
                false_positive=row["FP"],
                false_negative=row["FN"],
                false_positive_cost=fp_cost,
                false_negative_cost=fn_cost
            )

            rows.append({
                "method": row["method"],
                "false_positive_cost": fp_cost,
                "false_negative_cost": fn_cost,
                "expected_cost": cost
            })

    return pd.DataFrame(rows)


def plot_cost_sensitivity(
    sensitivity_df,
    output_path="figures/cost_sensitivity.png"
):
    plt.figure(figsize=(9, 6))

    for method in sensitivity_df["method"].unique():

        method_df = sensitivity_df[
            sensitivity_df["method"] == method
        ].sort_values(
            "false_negative_cost"
        )

        plt.plot(
            method_df["false_negative_cost"],
            method_df["expected_cost"],
            marker="o",
            label=method
        )

    # Break-even point between quantile_99 and median_std
    break_even = 33 / 9

    plt.axvline(
        break_even,
        linestyle="--",
        linewidth=1.5,
        label=f"Break-even ≈ {break_even:.2f}"
    )

    plt.xlabel(
        "Relative Cost of a Missed Anomaly"
    )

    plt.ylabel(
        "Expected Decision Cost"
    )

    plt.title(
        "Decision Cost Sensitivity Across Threshold Methods"
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":

    sensitivity_df = run_cost_sensitivity(
    fn_cost_values=[
        1,
        2,
        3,
        3.5,
        3.67,
        4,
        5,
        10,
        20
    ],
    fp_cost=1.0)

    print(sensitivity_df)

    sensitivity_df.to_csv(
        "data/cost_sensitivity.csv",
        index=False
    )

    plot_cost_sensitivity(
        sensitivity_df
    )