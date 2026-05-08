"""
Preprocessing module for sheep data cleaning and transformation.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load 3 tabel dari Excel.

    Returns:
        df_sheep, df_weight, df_health
    """
    df_sheep  = pd.read_excel(filepath, sheet_name="sheep")
    df_weight = pd.read_excel(filepath, sheet_name="weight_records")
    df_health = pd.read_excel(filepath, sheet_name="health_records")

    return df_sheep, df_weight, df_health


def calculate_days_old(
    df_weight: pd.DataFrame,
    df_sheep: pd.DataFrame
) -> pd.DataFrame:
    """
    Hitung umur domba saat ditimbang (dalam hari).

    Karena DB tidak menyimpan days_old, dihitung dari:
        days_old = recorded_at - birth_date
    """
    df_sheep  = df_sheep.copy()
    df_weight = df_weight.copy()

    df_sheep["birth_date"]   = pd.to_datetime(df_sheep["birth_date"])
    df_weight["recorded_at"] = pd.to_datetime(df_weight["recorded_at"])

    df_weight = df_weight.merge(
        df_sheep[["id", "birth_date"]].rename(columns={"id": "sheep_id"}),
        on="sheep_id",
        how="left"
    )

    df_weight["days_old"] = (
        df_weight["recorded_at"].dt.normalize() -
        df_weight["birth_date"].dt.normalize()
    ).dt.days

    return df_weight


def pivot_weight(df_weight: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot weight_records menjadi 1 baris per domba.
    Tiap titik ukur jadi kolom sendiri.

    Toleransi ±7 hari karena penimbangan di lapangan
    tidak selalu tepat di hari target.
    """
    CHECKPOINTS = {
        "weight_birth"   : (0,   7),
        "weight_weaning" : (90,  7),
        "weight_180d"    : (180, 7),
        "weight_365d"    : (365, 7),
    }

    result = df_weight[["sheep_id"]].drop_duplicates().copy()

    for col_name, (target_day, tol) in CHECKPOINTS.items():
        mask = (
            (df_weight["days_old"] >= target_day - tol) &
            (df_weight["days_old"] <= target_day + tol)
        )
        subset = (
            df_weight[mask]
            .groupby("sheep_id")["weight"]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={"weight": col_name})
        )
        result = result.merge(subset, on="sheep_id", how="left")

    return result


def calculate_adg(weight_features: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung ADG (Average Daily Gain) dari kolom berat.

    ADG_0_90   = (weight_weaning - weight_birth) / 90
    ADG_90_180 = (weight_180d - weight_weaning) / 90
    """
    df = weight_features.copy()

    df["ADG_0_90"] = (
        (df["weight_weaning"] - df["weight_birth"]) / 90
    ).round(4)

    df["ADG_90_180"] = (
        (df["weight_180d"] - df["weight_weaning"]) / 90
    ).round(4)

    return df


def calculate_health_score(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung health_score per domba dari health_records.

    Rumus: 1 - (total_severity / (total_events x 3))

    severity : ringan=1, sedang=2, berat=3, normal=0
    Skor 1.0 = tidak pernah sakit
    Skor 0.0 = selalu sakit parah
    """
    SEVERITY_WEIGHT = {"ringan": 1, "sedang": 2, "berat": 3, "normal": 0}
    SEVERITY_MAX    = 3

    df = df_health.copy()
    df["severity_score"] = df["severity"].map(SEVERITY_WEIGHT).fillna(0)

    health_agg = df.groupby("sheep_id").agg(
        total_events   = ("id",             "count"),
        total_severity = ("severity_score", "sum"),
    ).reset_index()

    health_agg["health_score"] = (
        1 - (health_agg["total_severity"] /
             (health_agg["total_events"] * SEVERITY_MAX + 1e-9))
    ).clip(0, 1).round(4)

    return health_agg[["sheep_id", "health_score"]]


def encode_categorical(df_sheep: pd.DataFrame) -> pd.DataFrame:
    """
    gender : "male" → 1, "female" → 0
    """
    df = df_sheep.copy()

    df["gender_enc"] = (df["gender"] == "male").astype(int)

    return df


def merge_all(
    df_sheep        : pd.DataFrame,
    weight_features : pd.DataFrame,
    health_features : pd.DataFrame,
) -> pd.DataFrame:
    """
    Gabungkan semua fitur menjadi 1 baris per domba.

    Urutan:
        1. Tabel sheep sebagai fondasi
        2. Join weight_features → berat & ADG
        3. Join health_features → health_score
    """
    df = df_sheep[[
        "id", "gender_enc", "breed_id", "sire_id", "dam_id"
    ]].rename(columns={"id": "sheep_id"})

    df = df.merge(weight_features, on="sheep_id", how="left")
    df = df.merge(health_features, on="sheep_id", how="left")

    df["health_score"] = df["health_score"].fillna(1.0)

    return df


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Simpan hasil preprocessing ke CSV.
    """
    df.to_csv(output_path, index=False)


def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Jalankan full pipeline preprocessing.

    Returns:
        df hasil preprocessing (sebelum disimpan)
    """
    df_sheep, df_weight, df_health = load_data(input_path)

    df_weight       = calculate_days_old(df_weight, df_sheep)
    weight_features = pivot_weight(df_weight)
    weight_features = calculate_adg(weight_features)
    health_features = calculate_health_score(df_health)
    df_sheep        = encode_categorical(df_sheep)
    df              = merge_all(df_sheep, weight_features, health_features)

    save_to_csv(df, output_path)

    return df
