import pandas as pd
import matplotlib.pyplot as plt


def load_results(
    results_path="data/monte_carlo_threshold_results.csv",
    summary_path="data/monte_carlo_threshold_summary.csv"
):
    results_df = pd.read_csv(results_path)
    summary_df = pd.read_csv(summary_path)

    return results_df, summary_df


def plot_fpr_fnr_tradeoff(
    summary_df,
    output_path="figures/fpr_fnr_tradeoff.png"
):
    plt.figure(figsize=(8, 6))

    label_offsets = {
        "mean_std": (0.002, 0.008),
        "median_std": (0.002, -0.002),
        "quantile_99": (0.002, 0.004),
        "median_mad": (0.002, 0.004)
    }

    for _, row in summary_df.iterrows():

        x = row["false_positive_rate_mean"]
        y = row["false_negative_rate_mean"]
        method = row["method"]

        plt.scatter(
            x,
            y,
            s=100
        )

        dx, dy = label_offsets.get(
            method,
            (0.002, 0.005)
        )

        plt.text(
            x + dx,
            y + dy,
            method
        )

    plt.xlabel(
        "Mean False Positive Rate"
    )

    plt.ylabel(
        "Mean False Negative Rate"
    )

    plt.title(
        "False Positive vs False Negative Trade-off"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_recall_distribution(
    results_df,
    output_path="figures/recall_distribution.png"
):
    methods = results_df["method"].unique()

    recall_data = [
        results_df[
            results_df["method"] == method
        ]["recall"].to_numpy()
        for method in methods
    ]

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        recall_data,
        labels=methods
    )

    plt.ylabel("Recall")
    plt.title(
        "Recall Stability Across Monte Carlo Calibration Samples"
    )
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_threshold_variability(
    results_df,
    output_path="figures/threshold_variability.png"
):
    methods = results_df["method"].unique()

    metric_columns = [
        "temperature_threshold",
        "vibration_threshold",
        "power_threshold"
    ]

    metric_labels = [
        "Temperature",
        "Vibration",
        "Power"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    for ax, column, label in zip(
        axes,
        metric_columns,
        metric_labels
    ):

        threshold_data = [
            results_df[
                results_df["method"] == method
            ][column].to_numpy()
            for method in methods
        ]

        ax.boxplot(
            threshold_data,
            labels=methods
        )

        ax.set_title(
            f"{label} Threshold"
        )

        ax.set_ylabel(
            "Estimated Threshold"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    fig.suptitle(
        "Threshold Variability Across Calibration Samples"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    results_df, summary_df = load_results()

    plot_fpr_fnr_tradeoff(
        summary_df
    )

    plot_recall_distribution(
        results_df
    )

    plot_threshold_variability(
        results_df
    )