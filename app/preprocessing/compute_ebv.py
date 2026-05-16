"""
EBV (Estimated Breeding Value) computation module.
"""

import pandas as pd
import numpy as np

H2 = {
    "bobot"    : 0.31,   # Nurjulaeha 2015, Domba Garut Margawati
    "adg"      : 0.28,   # estimasi konservatif, kisaran literatur
    "kesehatan": 0.10,   # rendah — kesehatan lebih dipengaruhi lingkungan
}


def compute_ebv_bobot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung EBV_Bobot dari weight_weaning dengan koreksi jenis kelamin.

    Koreksi gender dilakukan sebelum perhitungan EBV karena jantan secara
    hormonal cenderung lebih berat dari betina, terlepas dari mutu genetik.
    Tanpa koreksi, EBV jantan akan sistematis positif dan betina negatif
    sehingga ranking menjadi bias jenis kelamin, bukan ranking genetik murni.

    Rumus:
        P_adj     = weight_weaning - mean(kelamin) + mean(populasi)
        EBV_Bobot = h² x (P_adj - mean(P_adj))

    Args:
        df: DataFrame hasil handle_imputation(). Harus mengandung kolom
            weight_weaning (kg, sudah diimputasi) dan gender_enc (0/1).

    Returns:
        DataFrame dengan kolom baru EBV_Bobot (float, dibulatkan 4 desimal).
        Nilai positif = potensi genetik bobot sapih di atas rata-rata populasi.
        Nilai negatif = potensi genetik bobot sapih di bawah rata-rata populasi.
    """
    df = df.copy()

    gender_mean = df.groupby("gender_enc")["weight_weaning"].transform("mean")
    pop_mean    = df["weight_weaning"].mean()
    adj         = df["weight_weaning"] - gender_mean + pop_mean

    df["EBV_Bobot"] = (H2["bobot"] * (adj - adj.mean())).round(4)

    return df


def compute_ebv_adg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung EBV_ADG dari ADG_0_90 dengan koreksi jenis kelamin.

    Koreksi gender dilakukan karena jantan cenderung memiliki laju
    pertumbuhan lebih tinggi dari betina secara hormonal, bukan karena
    keunggulan genetik. Tanpa koreksi, ranking ADG akan bias jenis
    kelamin.

    Rumus:
        P_adj   = ADG_0_90 - mean(kelamin) + mean(populasi)
        EBV_ADG = h² x (P_adj - mean(P_adj))

    Args:
        df: DataFrame hasil handle_imputation(). Harus mengandung kolom
            ADG_0_90 (kg/hari, sudah diimputasi) dan gender_enc (0/1).

    Returns:
        DataFrame dengan kolom baru EBV_ADG (float, dibulatkan 4 desimal).
        Nilai positif = laju pertumbuhan harian di atas rata-rata genetik populasi.
        Nilai negatif = laju pertumbuhan harian di bawah rata-rata genetik populasi.
    """
    df = df.copy()

    gender_mean = df.groupby("gender_enc")["ADG_0_90"].transform("mean")
    pop_mean    = df["ADG_0_90"].mean()
    adj         = df["ADG_0_90"] - gender_mean + pop_mean

    df["EBV_ADG"] = (H2["adg"] * (adj - adj.mean())).round(4)

    return df


def compute_ebv_kesehatan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung EBV_Kesehatan dari health_score.
    Rumus:
        EBV_Kesehatan = h² x (health_score - mean_populasi)

    Args:
        df: DataFrame hasil merge_all() / handle_imputation(). Harus
            mengandung kolom health_score (float 0.0-1.0, tidak ada missing
            karena domba tanpa rekam penyakit sudah diisi 1.0 di preprocessing).

    Returns:
        DataFrame dengan kolom baru EBV_Kesehatan (float, dibulatkan 4 desimal).
        Nilai positif = ketahanan kesehatan genetik di atas rata-rata populasi.
        Nilai negatif = ketahanan kesehatan genetik di bawah rata-rata populasi.
    """
    df = df.copy()

    pop_mean = df["health_score"].mean()

    df["EBV_Kesehatan"] = (H2["kesehatan"] * (df["health_score"] - pop_mean)).round(4)

    return df


def add_parent_ebv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kontribusi Parent Average ke tiap EBV.

    Rumus:
        EBV_final = EBV_own + 0.5 × (EBV_sire + EBV_dam)

    Referensi:
        Wright, S. (1922). Coefficient of inbreeding and relationship.
        American Naturalist, 56, 330-338.

    Kasus:
        Sire + Dam ada  → 0.5 × (EBV_sire + EBV_dam)
        Hanya satu ada  → 0.5 × EBV_yang_ada
        Keduanya NULL   → 0.0 (founder Gen 0)
    """
    df = df.copy()

    ebv_lookup = df.set_index("sheep_id")[
        ["EBV_Bobot", "EBV_ADG", "EBV_Kesehatan"]
    ].to_dict(orient="index")

    for trait in ["EBV_Bobot", "EBV_ADG", "EBV_Kesehatan"]:
        parent_avg = []

        for _, row in df.iterrows():
            sire_id = row["sire_id"]
            dam_id  = row["dam_id"]

            sire_ebv = ebv_lookup.get(sire_id, {}).get(trait, 0.0) if pd.notna(sire_id) else 0.0
            dam_ebv  = ebv_lookup.get(dam_id,  {}).get(trait, 0.0) if pd.notna(dam_id)  else 0.0

            parent_avg.append(0.5 * (sire_ebv + dam_ebv))

        df[trait] = (df[trait] + pd.Series(parent_avg)).round(4)

    return df

def compute_all_ebv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: DataFrame hasil preprocessing + imputasi.
            Harus mengandung kolom:
            weight_weaning, ADG_0_90, health_score,
            sire_id, dam_id, sheep_id

    Returns:
        DataFrame dengan tambahan kolom:
        EBV_Bobot, EBV_ADG, EBV_Kesehatan
    """
    df = compute_ebv_bobot(df)
    df = compute_ebv_adg(df)
    df = compute_ebv_kesehatan(df)
    df = add_parent_ebv(df)

    return df
