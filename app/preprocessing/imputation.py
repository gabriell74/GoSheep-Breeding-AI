import pandas as pd
import numpy as np
from compute_ebv import load_preprocessed

def handle_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputasi hanya untuk kolom yang missing karena
    data collection loss (bukan karena belum waktunya).

    Strategi:
    - weight_weaning, ADG_0_90 → Group Mean (breed_id + gender_enc)
      Alasan: domba harusnya sudah pernah disapih,
              datanya hilang karena missing 12% di lapangan

    - weight_180d, weight_365d, ADG_90_180 → BIARKAN NULL
      Alasan: domba muda memang belum waktunya,
              bukan data hilang tapi belum terjadi

    - sire_id, dam_id → JANGAN diimputasi
      Alasan: NULL = founder Gen 0, memang tidak ada orang tua
    """
    df = df.copy()

    IMPUTE_COLS = ["weight_weaning", "ADG_0_90"]

    print(f"\n[handle_imputation] Null SEBELUM imputasi:")
    for col in IMPUTE_COLS:
        print(f"    {col:<20} null: {df[col].isna().sum()}")

    for col in IMPUTE_COLS:
        group_mean = df.groupby(["breed_id", "gender_enc"])[col].transform("mean")

        df[col] = df[col].fillna(group_mean)

    print(f"\n[handle_imputation] Null SESUDAH imputasi:")
    for col in IMPUTE_COLS:
        print(f"    {col:<20} null: {df[col].isna().sum()}")

    print(f"\n[handle_imputation] Kolom dibiarkan NULL (belum waktunya):")
    for col in ["weight_180d", "weight_365d", "ADG_90_180"]:
        print(f"    {col:<20} null: {df[col].isna().sum()}")

    return df

if __name__ == "__main__":
    df = load_preprocessed("data/processed/preprocessed.csv")
    df = handle_imputation(df)
