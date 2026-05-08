"""
Shared fixtures untuk semua test preprocessing & imputation.
Data dummy kecil — tidak bergantung pada file Excel asli.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_sheep() -> pd.DataFrame:
    """
    10 domba dummy dengan variasi:
    - 2 breed (1=Garut, 2=DEG)
    - 2 gender
    - 2 generasi (ada & tidak ada sire/dam)
    """
    return pd.DataFrame({
        "id"        : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gender"    : ["male","female","male","female","male",
                       "female","male","female","male","female"],
        "breed_id"  : [1, 1, 1, 1, 2, 2, 2, 2, 1, 2],
        "sire_id"   : [None, None, 1, 1, None, None, 5, 5, 1, 5],
        "dam_id"    : [None, None, 2, 2, None, None, 6, 6, 2, 6],
        "birth_date": [
            "2018-01-01","2018-02-01","2020-03-01","2020-04-01",
            "2018-05-01","2018-06-01","2021-01-01","2021-02-01",
            "2022-06-01","2022-07-01",
        ],
        "cage_id"   : [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "status"    : ["active"] * 10,
    })


@pytest.fixture
def sample_weight_records(sample_sheep) -> pd.DataFrame:
    """
    Weight records untuk 10 domba.
    - Domba 1–8: punya data hari ke 0, 90, 180, 365
    - Domba 9–10: hanya punya data hari ke 0
                  (simulasi domba muda / missing)
    """
    rows = []
    WID  = 1

    data = {
        # id : {days: weight}
        1 : {0: 2.87, 90: 12.52, 180: 16.40, 365: 22.10},
        2 : {0: 2.67, 90: 11.96, 180: 15.80, 365: 20.50},
        3 : {0: 2.90, 90: 13.10, 180: 17.00, 365: 23.00},
        4 : {0: 2.55, 90: 11.50, 180: 15.20, 365: 19.80},
        5 : {0: 2.70, 90: 10.11, 180: 13.50, 365: 18.00},
        6 : {0: 2.50, 90:  9.20, 180: 12.40, 365: 16.50},
        7 : {0: 2.80, 90: 12.00, 180: 16.00, 365: 21.50},
        8 : {0: 2.60, 90: 11.00, 180: 14.80, 365: 19.50},
        # Domba 9–10: hanya lahir (simulasi muda / missing)
        9 : {0: 2.75},
        10: {0: 2.45},
    }

    for sheep_id, checkpoints in data.items():
        birth_str = sample_sheep.loc[
            sample_sheep["id"] == sheep_id, "birth_date"
        ].values[0]
        birth = datetime.strptime(birth_str, "%Y-%m-%d")

        for days, weight in checkpoints.items():
            rows.append({
                "id"         : WID,
                "sheep_id"   : sheep_id,
                "weight"     : weight,
                "recorded_by": 1,
                "recorded_at": (birth + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"),
            })
            WID += 1

    return pd.DataFrame(rows)


@pytest.fixture
def sample_health_records() -> pd.DataFrame:
    """
    Health records untuk 6 dari 10 domba.
    Domba 9–10 tidak ada health_records
    (simulasi belum pernah sakit → health_score = 1.0)
    """
    return pd.DataFrame({
        "id"        : [1, 2, 3, 4, 5, 6, 7, 8],
        "sheep_id"  : [1, 1, 2, 3, 4, 5, 6, 7],
        "severity"  : ["ringan","sedang","berat","ringan","normal","sedang","ringan","berat"],
        "condition" : ["cacingan","pink_eye","cacingan","kembung","normal","cacingan","pink_eye","pneumonia"],
        "recorded_at": ["2020-01-01"] * 8,
    })


@pytest.fixture
def sample_preprocessed() -> pd.DataFrame:
    """
    Hasil preprocessing yang sudah di-merge (input untuk imputation & EBV).
    Ada NULL yang disengaja untuk test imputasi.
    """
    return pd.DataFrame({
        "sheep_id"      : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gender_enc"    : [1, 0, 1, 0, 1, 0, 1, 0, 1,  0],
        "breed_id"      : [1, 1, 1, 1, 2, 2, 2, 2, 1,  2],
        "sire_id"       : [None, None, 1.0, 1.0, None, None, 5.0, 5.0, 1.0, 5.0],
        "dam_id"        : [None, None, 2.0, 2.0, None, None, 6.0, 6.0, 2.0, 6.0],
        "weight_birth"  : [2.87, 2.67, 2.90, 2.55, 2.70, 2.50, 2.80, 2.60, 2.75, 2.45],
        "weight_weaning": [12.52, 11.96, 13.10, 11.50, 10.11, 9.20, 12.00, 11.00, None, None],
        "weight_180d"   : [16.40, 15.80, 17.00, 15.20, 13.50, 12.40, 16.00, 14.80, None, None],
        "weight_365d"   : [22.10, 20.50, 23.00, 19.80, 18.00, 16.50, 21.50, 19.50, None, None],
        "ADG_0_90"      : [0.1072, 0.1032, 0.1137, 0.0994, 0.0823, 0.0744, 0.1022, 0.0933, None, None],
        "ADG_90_180"    : [0.0431, 0.0427, 0.0433, 0.0411, 0.0376, 0.0356, 0.0444, 0.0422, None, None],
        "health_score"  : [0.67, 0.33, 0.67, 1.00, 0.67, 1.00, 0.67, 0.33, 1.00, 1.00],
    })
