import pandas as pd
import numpy as np
from imputation import handle_imputation

H2 = {
    "bobot" : 0.31, # Nurjulaeha 2015, Domba Garut Margawati
    "adg"   : 0.28,
    "kesehatan": 0.10, # rendah — kesehatan lebih dipengaruhi lingkungan
}

def load_preprocessed(filepath: str) -> pd.DataFrame:
    """
    Load hasil preprocessing
    """
    df = pd.read_csv(filepath)

    print(f"[load_preprocessed] {len(df)} domba, {len(df.columns)} kolom")

    return df

def compute_ebv_bobot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung EBV_Bobot dari weight_weaning.

    Rumus:
        EBV_Bobot = h² * (weight_weaning - mean_populasi)

    Nilai positif = di atas rata-rata genetik populasi
    Nilai negatif = di bawah rata-rata genetik populasi
    """
    df = df.copy()

    pop_mean = df["weight_weaning"].mean()

    df["EBV_Bobot"] = (
        H2["bobot"] * (df["weight_weaning"] - pop_mean)
    ).round(4)

    return df

if __name__ == "__main__":
    df = load_preprocessed("data/processed/preprocessed.csv")
    df = handle_imputation(df)
    df = compute_ebv_bobot(df)
