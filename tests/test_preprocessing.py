"""
Tests untuk preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np
from app.preprocessing.preprocessing import (
    calculate_days_old,
    pivot_weight,
    calculate_adg,
    calculate_health_score,
    encode_categorical,
    merge_all,
)


class TestCalculateDaysOld:

    def test_days_old_correct(self, sample_sheep, sample_weight_records):
        """days_old harus sesuai selisih recorded_at - birth_date."""
        result = calculate_days_old(sample_weight_records, sample_sheep)
        assert "days_old" in result.columns

    def test_days_old_non_negative(self, sample_sheep, sample_weight_records):
        """days_old tidak boleh negatif."""
        result = calculate_days_old(sample_weight_records, sample_sheep)
        assert (result["days_old"] >= 0).all()

    def test_days_old_birth_is_zero(self, sample_sheep, sample_weight_records):
        """Penimbangan di hari lahir harus menghasilkan days_old = 0."""
        result = calculate_days_old(sample_weight_records, sample_sheep)
        birth_records = result[result["days_old"] == 0]
        assert len(birth_records) > 0

    def test_original_not_mutated(self, sample_sheep, sample_weight_records):
        """Fungsi tidak boleh mengubah DataFrame input asli."""
        original_cols = list(sample_weight_records.columns)
        calculate_days_old(sample_weight_records, sample_sheep)
        assert list(sample_weight_records.columns) == original_cols


class TestPivotWeight:

    def test_output_columns(self, sample_sheep, sample_weight_records):
        """Output harus punya 5 kolom: sheep_id + 4 titik ukur."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        result = pivot_weight(df_w)
        expected_cols = ["sheep_id","weight_birth","weight_weaning","weight_180d","weight_365d"]
        assert list(result.columns) == expected_cols

    def test_one_row_per_sheep(self, sample_sheep, sample_weight_records):
        """Setiap domba hanya boleh punya 1 baris."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        result = pivot_weight(df_w)
        assert result["sheep_id"].nunique() == len(result)

    def test_weight_birth_no_null(self, sample_sheep, sample_weight_records):
        """weight_birth tidak boleh NULL karena semua domba punya data lahir."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        result = pivot_weight(df_w)
        assert result["weight_birth"].isna().sum() == 0

    def test_young_sheep_weaning_is_null(self, sample_sheep, sample_weight_records):
        """Domba muda (id 9,10) yang tidak punya data sapih harus NULL."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        result = pivot_weight(df_w)
        young  = result[result["sheep_id"].isin([9, 10])]
        assert young["weight_weaning"].isna().all()

    def test_weight_values_positive(self, sample_sheep, sample_weight_records):
        """Semua nilai berat yang ada harus positif."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        result = pivot_weight(df_w)
        for col in ["weight_birth","weight_weaning","weight_180d","weight_365d"]:
            vals = result[col].dropna()
            assert (vals > 0).all()


class TestCalculateAdg:

    def test_adg_columns_exist(self, sample_sheep, sample_weight_records):
        """Kolom ADG_0_90 dan ADG_90_180 harus ada di output."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        wf     = pivot_weight(df_w)
        result = calculate_adg(wf)
        assert "ADG_0_90"   in result.columns
        assert "ADG_90_180" in result.columns

    def test_adg_formula_correct(self, sample_preprocessed):
        """ADG_0_90 = (weight_weaning - weight_birth) / 90."""
        row     = sample_preprocessed.iloc[0]
        expected = round((row["weight_weaning"] - row["weight_birth"]) / 90, 4)
        result   = calculate_adg(sample_preprocessed)
        assert result.iloc[0]["ADG_0_90"] == expected

    def test_adg_null_when_weight_null(self, sample_preprocessed):
        """ADG harus NULL kalau weight komponennya NULL (domba muda)."""
        result = calculate_adg(sample_preprocessed)
        young  = result[result["sheep_id"].isin([9, 10])]
        assert young["ADG_0_90"].isna().all()

    def test_adg_positive_for_growing_sheep(self, sample_preprocessed):
        """ADG harus positif untuk domba yang tumbuh normal."""
        result  = calculate_adg(sample_preprocessed)
        adults  = result[result["sheep_id"].isin([1,2,3,4,5,6,7,8])]
        assert (adults["ADG_0_90"].dropna() > 0).all()


class TestCalculateHealthScore:

    def test_output_columns(self, sample_health_records):
        """Output harus punya kolom sheep_id dan health_score."""
        result = calculate_health_score(sample_health_records)
        assert "sheep_id"     in result.columns
        assert "health_score" in result.columns

    def test_score_range(self, sample_health_records):
        """health_score harus antara 0.0 dan 1.0."""
        result = calculate_health_score(sample_health_records)
        assert (result["health_score"] >= 0.0).all()
        assert (result["health_score"] <= 1.0).all()

    def test_normal_condition_not_penalized(self):
        """Kondisi 'normal' tidak boleh menurunkan skor."""
        df = pd.DataFrame({
            "id"      : [1, 2],
            "sheep_id": [1, 1],
            "severity": ["normal", "normal"],
        })
        result = calculate_health_score(df)
        assert result.iloc[0]["health_score"] == 1.0

    def test_severe_condition_lowers_score(self):
        """Kondisi berat harus menghasilkan skor lebih rendah dari ringan."""
        df_ringan = pd.DataFrame({"id":[1],"sheep_id":[1],"severity":["ringan"]})
        df_berat  = pd.DataFrame({"id":[1],"sheep_id":[1],"severity":["berat"]})

        score_ringan = calculate_health_score(df_ringan).iloc[0]["health_score"]
        score_berat  = calculate_health_score(df_berat).iloc[0]["health_score"]

        assert score_ringan > score_berat


class TestEncodeCategorical:

    def test_male_encoded_as_one(self, sample_sheep):
        """Jantan harus di-encode sebagai 1."""
        result = encode_categorical(sample_sheep)
        males  = result[result["gender"] == "male"]
        assert (males["gender_enc"] == 1).all()

    def test_female_encoded_as_zero(self, sample_sheep):
        """Betina harus di-encode sebagai 0."""
        result  = encode_categorical(sample_sheep)
        females = result[result["gender"] == "female"]
        assert (females["gender_enc"] == 0).all()

    def test_no_other_values(self, sample_sheep):
        """gender_enc hanya boleh berisi 0 atau 1."""
        result = encode_categorical(sample_sheep)
        assert set(result["gender_enc"].unique()).issubset({0, 1})

    def test_original_not_mutated(self, sample_sheep):
        """Fungsi tidak boleh mengubah DataFrame input asli."""
        encode_categorical(sample_sheep)
        assert "gender_enc" not in sample_sheep.columns


class TestMergeAll:

    def test_output_shape(self, sample_sheep, sample_weight_records, sample_health_records):
        """Output harus punya 1 baris per domba."""
        df_w    = calculate_days_old(sample_weight_records, sample_sheep)
        wf      = calculate_adg(pivot_weight(df_w))
        hf      = calculate_health_score(sample_health_records)
        df_s    = encode_categorical(sample_sheep)
        result  = merge_all(df_s, wf, hf)
        assert len(result) == len(sample_sheep)

    def test_required_columns_exist(self, sample_sheep, sample_weight_records, sample_health_records):
        """Kolom wajib harus ada di output."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        wf     = calculate_adg(pivot_weight(df_w))
        hf     = calculate_health_score(sample_health_records)
        df_s   = encode_categorical(sample_sheep)
        result = merge_all(df_s, wf, hf)

        for col in ["sheep_id","gender_enc","breed_id","weight_birth",
                    "weight_weaning","ADG_0_90","health_score"]:
            assert col in result.columns

    def test_no_health_records_fills_one(self, sample_sheep, sample_weight_records, sample_health_records):
        """Domba tanpa health_records harus dapat health_score = 1.0."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        wf     = calculate_adg(pivot_weight(df_w))
        hf     = calculate_health_score(sample_health_records)
        df_s   = encode_categorical(sample_sheep)
        result = merge_all(df_s, wf, hf)

        no_health = result[result["sheep_id"].isin([9, 10])]
        assert (no_health["health_score"] == 1.0).all()

    def test_sire_dam_null_preserved(self, sample_sheep, sample_weight_records, sample_health_records):
        """sire_id & dam_id NULL untuk founders harus tetap NULL."""
        df_w   = calculate_days_old(sample_weight_records, sample_sheep)
        wf     = calculate_adg(pivot_weight(df_w))
        hf     = calculate_health_score(sample_health_records)
        df_s   = encode_categorical(sample_sheep)
        result = merge_all(df_s, wf, hf)

        founders = result[result["sheep_id"].isin([1, 2, 5, 6])]
        assert founders["sire_id"].isna().all()
        assert founders["dam_id"].isna().all()
