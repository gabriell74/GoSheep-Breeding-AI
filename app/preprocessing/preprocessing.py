"""
Preprocessing module for data cleaning and transformation.
"""

import pandas as pd
import numpy as np

def load_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_sheep = pd.read_excel(filepath, sheet_name='sheep')
    df_weight = pd.read_excel(filepath, sheet_name='weight_records')
    df_health = pd.read_excel(filepath, sheet_name='health_records')

    print(f"[load_data] sheep : {len(df_sheep)}")
    print(f"[load_data] weight records : {len(df_weight)}")
    print(f"[load_data] health records : {len(df_health)}")

    return df_sheep, df_weight, df_health

def calculate_days_old(df_weight: pd.DataFrame, df_sheep: pd.DataFrame) -> pd.DataFrame:
    """Kalkulasi umur domba berdasarkan tanggal lahir dan tanggal pencatatan berat."""

    df_sheep = df_sheep.copy()
    df_weight = df_weight.copy()

    df_sheep['birth_date'] = pd.to_datetime(df_sheep['birth_date'])
    df_weight['recorded_at'] = pd.to_datetime(df_weight['recorded_at'])

    df_weight = df_weight.merge(
        df_sheep[['id', 'birth_date']].rename(columns={'id': 'sheep_id'}),
        on='sheep_id',
        how='left'
    )

    df_weight['days_old'] = (
        df_weight['recorded_at'].dt.normalize() -
        df_weight['birth_date'].dt.normalize()
    ).dt.days

    print(f"[calculate_days_old] Titik ukur ditemukan: {sorted(df_weight['days_old'].unique())}")

    return df_weight

def pivot_weight(df_weight: pd.DataFrame) -> pd.DataFrame:
  """
  Pivot weight_records untuk memiliki satu baris per domba dan kolom untuk setiap catatan berat.
  Tiap titik ukur menjadi kolom sendiri.
  Toleransi ±7 hari karena weight_records tidak selalu pada hari yang sama untuk setiap domba.
  """
  CHECKPOINTS = {
    "weight_birth": (0, 7),
    "weight_weaning": (90, 7),
    "weight_180d": (180, 7),
    "weight_365d": (365, 7)
  }

  result = df_weight[['sheep_id']].drop_duplicates().copy()

  for col_name, (target_day, tolerance) in CHECKPOINTS.items():
      mask = (
          (df_weight['days_old'] >= target_day - tolerance) &
          (df_weight['days_old'] <= target_day + tolerance)
      )
      subset = (
          df_weight[mask]
          .groupby('sheep_id')['weight']
          .mean()
          .round(2)
          .reset_index()
          .rename(columns={'weight': col_name})
      )
      result = result.merge(subset, on='sheep_id', how='left')

  print(f"[pivot_weight] Kolom: {list(result.columns)}")
  for col in result.columns[1:]:
      print(f"[pivot_weight]  {col:<20} null: {result[col].isna().sum()}")

  return result

def calculate_adg(weight_features: pd.DataFrame) -> pd.DataFrame:
    """
    Kalkulasi Average Daily Gain (ADG) dari kolom berat.
    ADG_0_90   = (weight_weaning - weight_birth) / 90
    ADG_90_180 = (weight_180d - weight_weaning) / 90
    """
    df = weight_features.copy()

    df["ADG_0_90"] = (
        (df['weight_weaning'] - df['weight_birth']) / 90
    ).round(4)

    df["ADG_90_180"] = (
        (df['weight_180d'] - df['weight_weaning']) / 90
    ).round(4)

    print(f"[calculate_adg] ADG_0_90   → mean: {df['ADG_0_90'].mean():.4f}  min: {df['ADG_0_90'].min():.4f}  max: {df['ADG_0_90'].max():.4f}")
    print(f"[calculate_adg] ADG_90_180 → mean: {df['ADG_90_180'].mean():.4f}  min: {df['ADG_90_180'].min():.4f}  max: {df['ADG_90_180'].max():.4f}")

    return df

def calculate_health_score(df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung health_score per domba dari health_records.

    Rumus: 1 - (total_severity / (total_events x 3))

    severity : ringan=1, sedang=2, berat=3
    Skor 1.0 = tidak pernah sakit
    Skor 0.0 = selalu sakit parah
    """
    SEVERITY_WEIGHT = {
        "ringan": 1,
        "sedang": 2,
        "berat": 3,
        "normal": 0
    }

    SEVERITY_MAX = 3

    df = df_health.copy()
    df["severity_score"] = df["severity"].map(SEVERITY_WEIGHT).fillna(0)

    health_agg = df.groupby("sheep_id").agg(
        total_events = ("id", "count"),
        total_severity = ("severity_score", "sum"),
    ).reset_index()

    health_agg["health_score"] = (
        1 - (health_agg["total_severity"] /
             (health_agg["total_events"] * SEVERITY_MAX + 1e-9))
    ).clip(0, 1).round(4)

    result = health_agg[["sheep_id", "health_score"]]
    print(f"[calculate_health_score] Domba dengan health_records: {len(result)}")
    print(f"[calculate_health_score] health_score → mean: {result['health_score'].mean():.2f}  min: {result['health_score'].min():.2f}  max: {result['health_score'].max():.2f}")

    return result

def encode_categorical(df_sheep: pd.DataFrame) -> pd.DataFrame:
    """
    Encode kolom kategorikal menjadi angka.

    gender : "male"   → 1
             "female" → 0
    """
    df = df_sheep.copy()

    df["gender_enc"] = (df["gender"] == "male").astype(int)

    print(f"[encode_categorical] Jantan (1): {(df['gender_enc'] == 1).sum()}")
    print(f"[encode_categorical] Betina (0): {(df['gender_enc'] == 0).sum()}")

    return df

def merge_all(df_sheep: pd.DataFrame,
              weight_features: pd.DataFrame,
              health_features: pd.DataFrame) -> pd.DataFrame:
     """
     Gabungkan semua fitur menjadi 1 baris per domba.

      Urutan:
      1. Mulai dari tabel sheep (fondasi)
      2. Join weight_features → tambah kolom berat & ADG
      3. Join health_features → tambah kolom health_score
     """

     df = df_sheep[[
          'id', 'gender_enc', 'breed_id', 'sire_id', 'dam_id'
     ]].rename(columns={'id': 'sheep_id'})

     df = df.merge(weight_features, on='sheep_id', how='left')

     df = df.merge(health_features, on='sheep_id', how='left')

     df['health_score'] = df['health_score'].fillna(1.0)

     print(f"[merge_all] Shape akhir: {df.shape[0]} baris × {df.shape[1]} kolom")
     print(f"[merge_all] Kolom: {list(df.columns)}")

     return df

def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Simpan hasil preprocessing ke CSV.
    """
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"HASIL AKHIR preprocessed.csv")
    print(f"{'='*50}")
    print(f"\n{'No':<4} {'Kolom':<22} {'Contoh':<15} {'Null'}")
    print(f"{'-'*50}")
    for i, col in enumerate(df.columns, 1):
        contoh = df[col].dropna().iloc[0] if df[col].notna().any() else "-"
        null   = df[col].isna().sum()
        if isinstance(contoh, float):
            contoh = round(contoh, 4)
        print(f"{i:<4} {col:<22} {str(contoh):<15} {null}")

    print(f"\n✅ Tersimpan di: {output_path}")
    print(f"   Total: {len(df)} domba, {len(df.columns)} kolom")

if __name__ == "__main__":
    filepath = "data/raw/gosheep_synthetic.xlsx"
    df_sheep, df_weight, df_health = load_data(filepath)

    df_weight = calculate_days_old(df_weight, df_sheep)
    weight_features = pivot_weight(df_weight)
    weight_features = calculate_adg(weight_features)
    health_features = calculate_health_score(df_health)
    df_sheep        = encode_categorical(df_sheep)
    df              = merge_all(df_sheep, weight_features, health_features)

    save_to_csv(df, "data/processed/preprocessed.csv")
