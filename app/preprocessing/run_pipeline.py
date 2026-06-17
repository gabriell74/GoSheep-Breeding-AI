"""
Pipeline lengkap dari raw data → training data.

Urutan:
    1. Load data dari Excel
    2. Preprocessing (pivot weight, ADG, health score)
    3. Imputasi missing values
    4. Compute EBV (Bobot, ADG, Kesehatan)
    5. Simpan training_data.csv
"""

import pandas as pd
from preprocessing import (
    load_data,
    calculate_days_old,
    pivot_weight,
    calculate_adg,
    calculate_health_score,
    encode_categorical,
    merge_all,
    save_to_csv,
)
from imputation import handle_imputation
from compute_ebv import compute_all_ebv


def run_pipeline(input_path: str,
                 preprocessed_path: str,
                 training_path: str) -> pd.DataFrame:
    """
    Jalankan full pipeline dari raw data sampai training data.

    Args:
        input_path        : path ke gosheep_synthetic.xlsx
        preprocessed_path : path output preprocessed.csv
        training_path     : path output training_data.csv

    Returns:
        df training yang sudah siap (dengan kolom EBV)
    """
    df_sheep, df_weight, df_health = load_data(input_path)

    df_weight       = calculate_days_old(df_weight, df_sheep)
    weight_features = pivot_weight(df_weight)
    weight_features = calculate_adg(weight_features)
    health_features = calculate_health_score(df_health)
    df_sheep        = encode_categorical(df_sheep)
    df              = merge_all(df_sheep, weight_features, health_features)

    save_to_csv(df, preprocessed_path)

    df = handle_imputation(df)

    df = compute_all_ebv(df)

    save_to_csv(df, training_path)

    return df

if __name__ == "__main__":
    df = run_pipeline(
        input_path        = "data/raw/gosheep_synthetic.xlsx",
        preprocessed_path = "data/processed/preprocessed.csv",
        training_path     = "data/processed/training_data.csv",
    )

    print(f"\n{'='*55}")
    print(f"HASIL AKHIR training_data.csv")
    print(f"{'='*55}")
    print(f"\n{'No':<4} {'Kolom':<22} {'Contoh':<15} {'Null'}")
    print(f"{'-'*55}")
    for i, col in enumerate(df.columns, 1):
        contoh = df[col].dropna().iloc[0] if df[col].notna().any() else "-"
        null   = df[col].isna().sum()
        if isinstance(contoh, float):
            contoh = round(contoh, 4)
        print(f"{i:<4} {col:<22} {str(contoh):<15} {null}")

    print(f"\n✅ Total: {len(df)} domba, {len(df.columns)} kolom")
    print(f"\nKolom EBV:")
    for col in ["EBV_Bobot", "EBV_ADG", "EBV_Kesehatan"]:
        print(f"  {col:<20} mean: {df[col].mean():.4f}  "
              f"min: {df[col].min():.4f}  "
              f"max: {df[col].max():.4f}")
