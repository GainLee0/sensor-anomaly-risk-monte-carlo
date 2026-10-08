import numpy as np
import pandas as pd


METRIC_NAMES = [
    "temperature",
    "vibration",
    "power"
]


def generate_sensor_data(
    n_normal=500,
    n_anomaly=100,
    random_seed=42
):
    """
    Generate fully synthetic IoT sensor observations.

    Ground-truth label:
        0 = normal
        1 = anomaly
    """

    rng = np.random.default_rng(random_seed)

    # --------------------------------
    # 1. Normal operating population
    # --------------------------------

    normal_mean = np.array([
        70.0,
        3.0,
        100.0
    ])

    normal_std = np.array([
        4.0,
        0.5,
        8.0
    ])

    correlation = np.array([
        [1.00, 0.40, 0.30],
        [0.40, 1.00, 0.35],
        [0.30, 0.35, 1.00]
    ])

    normal_cov = (
        np.outer(normal_std, normal_std)
        * correlation
    )

    normal_data = rng.multivariate_normal(
        mean=normal_mean,
        cov=normal_cov,
        size=n_normal
    )

    # --------------------------------
    # 2. Anomalous operating population
    # --------------------------------

    anomaly_mean = np.array([
        79.0,
        4.2,
        118.0
    ])

    anomaly_std = np.array([
        5.0,
        0.7,
        10.0
    ])

    anomaly_cov = (
        np.outer(anomaly_std, anomaly_std)
        * correlation
    )

    anomaly_data = rng.multivariate_normal(
        mean=anomaly_mean,
        cov=anomaly_cov,
        size=n_anomaly
    )

    # --------------------------------
    # 3. Build DataFrames
    # --------------------------------

    normal_df = pd.DataFrame(
        normal_data,
        columns=METRIC_NAMES
    )
    normal_df["is_anomaly"] = 0

    anomaly_df = pd.DataFrame(
        anomaly_data,
        columns=METRIC_NAMES
    )
    anomaly_df["is_anomaly"] = 1

    # --------------------------------
    # 4. Combine and shuffle
    # --------------------------------

    df = pd.concat(
        [normal_df, anomaly_df],
        ignore_index=True
    )

    df = df.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    df.insert(
        0,
        "observation_id",
        [
            f"OBS_{i+1:04d}"
            for i in range(len(df))
        ]
    )

    return df


if __name__ == "__main__":

    df = generate_sensor_data()

    df.to_csv(
        "data/synthetic_sensor_data.csv",
        index=False
    )

    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass counts:")
    print(df["is_anomaly"].value_counts())

    print("\nMetric correlations:")
    print(df[METRIC_NAMES].corr().round(2))