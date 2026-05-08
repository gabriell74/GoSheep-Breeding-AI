"""
Tests untuk imputation.py
"""

import pytest
import pandas as pd
import numpy as np
from app.preprocessing.imputation import handle_imputation


class TestHandleImputation:

    def test_weight_weaning_no_null_after(self, sample_preprocessed):
        """weight_weaning tidak boleh NULL setelah imputasi."""
        result = handle_imputation(sample_preprocessed)
        assert result["weight_weaning"].isna().sum() == 0

    def test_adg_0_90_no_null_after(self, sample_preprocessed):
        """ADG_0_90 tidak boleh NULL setelah imputasi."""
        result = handle_imputation(sample_preprocessed)
        assert result["ADG_0_90"].isna().sum() == 0

    def test_weight_180d_still_null(self, sample_preprocessed):
        """weight_180d harus tetap NULL (domba muda belum waktunya)."""
        null_before = sample_preprocessed["weight_180d"].isna().sum()
        result      = handle_imputation(sample_preprocessed)
        assert result["weight_180d"].isna().sum() == null_before

    def test_weight_365d_still_null(self, sample_preprocessed):
        """weight_365d harus tetap NULL (domba muda belum waktunya)."""
        null_before = sample_preprocessed["weight_365d"].isna().sum()
        result      = handle_imputation(sample_preprocessed)
        assert result["weight_365d"].isna().sum() == null_before

    def test_adg_90_180_still_null(self, sample_preprocessed):
        """ADG_90_180 harus tetap NULL (domba muda belum waktunya)."""
        null_before = sample_preprocessed["ADG_90_180"].isna().sum()
        result      = handle_imputation(sample_preprocessed)
        assert result["ADG_90_180"].isna().sum() == null_before

    def test_sire_id_not_imputed(self, sample_preprocessed):
        """sire_id NULL untuk founders tidak boleh diimputasi."""
        null_before = sample_preprocessed["sire_id"].isna().sum()
        result      = handle_imputation(sample_preprocessed)
        assert result["sire_id"].isna().sum() == null_before

    def test_dam_id_not_imputed(self, sample_preprocessed):
        """dam_id NULL untuk founders tidak boleh diimputasi."""
        null_before = sample_preprocessed["dam_id"].isna().sum()
        result      = handle_imputation(sample_preprocessed)
        assert result["dam_id"].isna().sum() == null_before

    def test_imputed_value_is_group_mean(self, sample_preprocessed):
        """
        Nilai imputasi harus sesuai group mean (breed_id + gender_enc).
        Domba 9: breed_id=1, gender_enc=1 (jantan Garut)
        Nilai harus = mean weight_weaning jantan Garut yang tidak NULL.
        """
        result = handle_imputation(sample_preprocessed)

        group_mean = (
            sample_preprocessed[
                (sample_preprocessed["breed_id"]   == 1) &
                (sample_preprocessed["gender_enc"] == 1) &
                (sample_preprocessed["weight_weaning"].notna())
            ]["weight_weaning"].mean()
        )

        sheep_9_value = result.loc[result["sheep_id"] == 9, "weight_weaning"].values[0]
        assert abs(sheep_9_value - group_mean) < 1e-6

    def test_existing_values_not_changed(self, sample_preprocessed):
        """Nilai yang sudah ada (tidak NULL) tidak boleh berubah."""
        result = handle_imputation(sample_preprocessed)

        original_not_null = sample_preprocessed["weight_weaning"].dropna()
        result_not_null   = result.loc[
            sample_preprocessed["weight_weaning"].notna(), "weight_weaning"
        ]
        pd.testing.assert_series_equal(
            original_not_null.reset_index(drop=True),
            result_not_null.reset_index(drop=True),
        )

    def test_original_not_mutated(self, sample_preprocessed):
        """Fungsi tidak boleh mengubah DataFrame input asli."""
        null_before = sample_preprocessed["weight_weaning"].isna().sum()
        handle_imputation(sample_preprocessed)
        assert sample_preprocessed["weight_weaning"].isna().sum() == null_before

    def test_imputed_values_positive(self, sample_preprocessed):
        """Nilai hasil imputasi harus positif (berat tidak boleh <= 0)."""
        result = handle_imputation(sample_preprocessed)
        assert (result["weight_weaning"] > 0).all()
        assert (result["ADG_0_90"] > 0).all()
