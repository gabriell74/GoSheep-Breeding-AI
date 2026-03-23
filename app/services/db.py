import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

load_dotenv()

# ── Build connection URL ────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "gosheep_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)

# ── Create engine ───────────────────────────────────────────
# pool_pre_ping=True → otomatis reconnect kalau koneksi putus
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Context manager untuk session ──────────────────────────
@contextmanager
def get_db() -> Session:
    """
    Digunakan jika membutuhkan session ORM.

    Contoh:
        with get_db() as db:
            result = db.execute(text("SELECT * FROM sheep"))
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Helper untuk pandas read_sql ───────────────────────────
def get_engine():
    """
    Digunakan untuk pandas read_sql.

    Contoh:
        import pandas as pd
        df = pd.read_sql("SELECT * FROM sheep", get_engine())
    """
    return engine


# ── Test koneksi ────────────────────────────────────────────
def test_connection() -> bool:
    """Cek apakah koneksi ke DB berhasil."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[DB] Koneksi berhasil ke {DB_HOST}:{DB_PORT}/{DB_NAME}")
        return True
    except Exception as e:
        print(f"[DB] Koneksi gagal: {e}")
        return False


if __name__ == "__main__":
    test_connection()
