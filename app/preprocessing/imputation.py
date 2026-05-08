"""
Imputation module for handling missing values in preprocessed sheep data.
"""

import pandas as pd


def handle_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputasi hanya untuk kolom yang missing karena data collection loss,
    bukan karena domba belum cukup umur.

    Strategi:
        - weight_weaning, ADG_0_90  → Group Mean (breed_id + gender_enc)
          Alasan: domba harusnya sudah pernah disapih,
                  datanya hilang karena missing ~12% di lapangan

        - weight_180d, weight_365d, ADG_90_180 → BIARKAN NULL
          Alasan: domba muda memang belum waktunya,
                  bukan data hilang tapi memang belum terjadi

        - sire_id, dam_id → JANGAN diimputasi
          Alasan: NULL = founder Gen 0, memang tidak ada orang tua
    """
    df = df.copy()

    IMPUTE_COLS = ["weight_weaning", "ADG_0_90"]

    for col in IMPUTE_COLS:
        group_mean = df.groupby(["breed_id", "gender_enc"])[col].transform("mean")
        df[col]    = df[col].fillna(group_mean)

    return df
