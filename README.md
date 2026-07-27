# 🐑 GoSheep Breeding AI — Smart Breeding Domba

Servis AI untuk rekomendasi perkawinan domba dan kalkulasi **Estimated Breeding Value (EBV)** menggunakan **Random Forest**, **Wright's Coefficient of Inbreeding**, **Analytic Hierarchy Process (AHP)**, dan **Multi-Objective Optimization on the basis of Ratio Analysis (MOORA)** untuk mengoptimalkan mutu genetik keturunan.

---

## 📌 Ringkasan Proyek

Servis ini dibuat menggunakan **FastAPI** yang berfungsi sebagai mesin rekomendasi perkawinan (_breeding recommendation engine_) untuk aplikasi GoSheep. Servis ini menerima data fenotip, silsilah (_pedigree_), dan kriteria pembobotan dari Backend (Laravel), kemudian menghasilkan:

- **Prediksi EBV** individu (`EBV_Bobot`, `EBV_ADG`, `EBV_Kesehatan`).
- **Filter Inbreeding** menggunakan Koefisien Wright ($F \ge 6.25\%$).
- **Ranking Pasangan Perkawinan Terbaik** berbasis AHP & MOORA.

---

## ⚙️ Alur & Metodologi Sistem

### 1. Fase Prediksi EBV (Saat Registrasi / Update Data Domba)

```
Data Fenotip + Silsilah → Preprocessing & Imputasi → Model Machine Learning (Random Forest) → Prediksi EBV
```

### 2. Fase Rekomendasi Perkawinan (Fase Query)

```
1. Pilih Domba (Betina / Ewe)
   └── Ambil kandidat lawan jenis (Jantan / Ram)
2. Wright's Inbreeding Coefficient Calculation
   └── Filter & eliminasi pasangan dengan F ≥ 6.25% (mencegah inbreeding depresif)
3. Expected EBV Offspring Calculation
   └── Estimasi EBV Anak = (EBV_Ewe + EBV_Ram) / 2
4. Analytic Hierarchy Process (AHP)
   └── Tentukan bobot prioritas kriteria (Bobot Lahir/Sapih, ADG, Kesehatan)
5. MOORA (Multi-Objective Optimization)
   └── Perhitungan matriks keputusan & kalkulasi nilai preferensi (ranking)
6. Output Top-N Pasangan Rekomendasi Terbaik
```

---

## 🛠️ Tech Stack & Persyaratan Sistem

### Stack Utama

- **Python** ^3.10 / ^3.11
- **FastAPI** (Framework Web REST API)
- **Uvicorn** (ASGI Web Server)
- **Scikit-Learn / XGBoost** (Machine Learning Engine)
- **Pandas & NumPy** (Manipulasi Data & Matriks)
- **Pydantic** (Validasi Schema Data & Request)
- **Pytest** (Framework Testing)

### Persyaratan Sistem

- Python 3.10 atau versi lebih baru
- `pip` (Python Package Installer)
- `virtualenv` / modul `venv` bawaan Python

---

## Panduan Instalasi dan Setup

### 1. Clone Repository

```bash
git clone https://github.com/gabriell74/GoSheep-Breeding-AI.git
cd GoSheep-Breeding-AI
```

### 2. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependensi Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Environment File (`.env`)

Salin file konfigurasi contoh ke `.env`:

**Windows:**

```cmd
copy .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Buka file `.env` dan sesuaikan variabel lingkungan jika diperlukan:

```ini
# App Configuration
APP_ENV=development
APP_PORT=8001

# Model Versioning
MODEL_VERSION=1.0.0
```

### 5. Memastikan Model Machine Learning Tersedia

Pastikan file model trained Random Forest (`ebv_model.pkl`) sudah ada di dalam folder `models/`:

```
models/
├── .gitkeep
└── ebv_model.pkl
```

---

## 🏃 Menjalankan Aplikasi

### Mode Development

Jalankan server FastAPI menggunakan Uvicorn:

```bash
uvicorn app.main:app --port 8001
```

Atau menggunakan modul Python secara langsung:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Server akan berjalan pada URL: `http://localhost:8001`

---

## 📄 Dokumentasi API (Interactive Docs)

FastAPI menyediakan dokumentasi interaktif secara otomatis yang dapat diakses di browser setelah server dinyalakan:

- **Swagger UI (Interactive API Docs)**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

---

## 🧪 Menjalankan Pengujian (Testing)

Untuk memastikan seluruh fungsi preprocessing, model machine learning, kalkulasi Wright, dan kalkulasi AHP/MOORA berjalan dengan benar:

```bash
pytest
```

Untuk melihat hasil output test secara mendalam:

```bash
pytest -v
```

---

## 📁 Struktur Singkat Proyek

```
gosheep_breeding_ai/
├── app/
│   ├── main.py             # Entry point FastAPI & registrasi router
│   ├── models/             # Schema Pydantic & struktur data
│   ├── preprocessing/      # Modul pembersihan data & imputasi
│   ├── routes/             # Endpoint REST API (predict, wright, ebv)
│   ├── schemas/            # Schemas request/response
│   └── services/           # Logika AI (Random Forest, Wright, AHP, MOORA)
├── data/                   # Dataset rujukan & file data contoh
├── models/                 # Pre-trained ML model binary (ebv_model.pkl)
├── tests/                  # Unit test & integration test (pytest)
├── .env.example            # Template file variabel lingkungan
├── pytest.ini              # Konfigurasi runner pytest
└── requirements.txt        # Daftar dependensi paket Python
```

---

## 🔗 Integrasi dengan Backend GoSheep (Laravel API)

Servis AI ini dikonsumsi secara langsung oleh **GoSheep API (Laravel Backend)**:

1. Pastikan servis AI ini berjalan di `http://localhost:8001`.
2. Di proyek **GoSheep API (Laravel)**, pastikan file `.env` menyertakan konfigurasi URL AI:
   ```ini
   GOSHEEP_AI_URL=http://localhost:8001
   ```
