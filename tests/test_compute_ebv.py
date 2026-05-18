import pandas as pd

from app.preprocessing.compute_ebv import (
    compute_ebv_bobot,
    compute_ebv_adg,
    compute_ebv_kesehatan,
    add_parent_ebv,
    compute_all_ebv,
)


def sample_df():
    return pd.DataFrame({
        "sheep_id": [1, 2, 3],
        "gender_enc": [0, 1, 0],
        "weight_weaning": [20.0, 30.0, 25.0],
        "ADG_0_90": [0.2, 0.3, 0.25],
        "health_score": [0.9, 0.8, 1.0],
        "sire_id": [None, 1, 1],
        "dam_id": [None, None, 2],
    })


def test_compute_ebv_bobot():
    df = sample_df()

    result = compute_ebv_bobot(df)

    assert "EBV_Bobot" in result.columns
    assert result["EBV_Bobot"].dtype == float


def test_compute_ebv_adg():
    df = sample_df()

    result = compute_ebv_adg(df)

    assert "EBV_ADG" in result.columns
    assert result["EBV_ADG"].dtype == float


def test_compute_ebv_kesehatan():
    df = sample_df()

    result = compute_ebv_kesehatan(df)

    assert "EBV_Kesehatan" in result.columns
    assert result["EBV_Kesehatan"].dtype == float


def test_add_parent_ebv():
    df = sample_df()

    df = compute_ebv_bobot(df)
    df = compute_ebv_adg(df)
    df = compute_ebv_kesehatan(df)

    result = add_parent_ebv(df)

    assert "EBV_Bobot" in result.columns
    assert "EBV_ADG" in result.columns
    assert "EBV_Kesehatan" in result.columns


def test_compute_all_ebv():
    df = sample_df()

    result = compute_all_ebv(df)

    assert "EBV_Bobot" in result.columns
    assert "EBV_ADG" in result.columns
    assert "EBV_Kesehatan" in result.columns

    assert len(result) == 3