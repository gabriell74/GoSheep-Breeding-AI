# 🐑 AI-Based Selective Mating — Smart Breeding Domba

Servis AI untuk rekomendasi perkawinan domba menggunakan **Random Forest**, **Wright Coefficient**, **AHP**, dan **MOORA** untuk mengoptimalkan mutu genetik keturunan.

---

## Alur Sistem

**Fase Registrasi** (saat domba didaftarkan):
```
Data domba (fenotip + silsilah) → Random Forest → EBV per domba → simpan ke DB
```

**Fase Query** (saat peternak mau breeding):
```
Pilih 1 domba → ambil kandidat lawan jenis → Wright Coefficient (filter F ≥ 6.25%)
→ Expected EBV Offspring = (EBV_Ewe + EBV_Ram) / 2 → AHP (bobot trait) → MOORA (ranking)
→ Top-N rekomendasi pasangan
```

---

## Komponen

| Komponen | Fungsi |
|---|---|
| **Random Forest** | Prediksi EBV individu dari fenotip + silsilah |
| **Wright Coefficient** | Filter pasangan dengan inbreeding F ≥ 6.25% |
| **Expected EBV** | Estimasi kualitas genetik keturunan |
| **AHP** | Pembobotan kepentingan antar trait |
| **MOORA** | Ranking pasangan berdasarkan skor akhir |

**Trait EBV:** `EBV_Bobot` · `EBV_ADG` · `EBV_Kesehatan`

---

## Tech Stack

`Python` · `FastAPI` · `scikit-learn / XGBoost` · `MySQL`

---

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
