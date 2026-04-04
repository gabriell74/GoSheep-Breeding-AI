# tests/test_wright.py
import sys
sys.path.insert(0, ".")

import pytest
from app.services.wright_service import (
    load_pedigree,
    get_candidates,
    filter_safe_candidates,
    calculate_coi,
    is_safe_pair,
    INBREEDING_THRESHOLD,
)

# Struktur:
#   1 (founder) + 2 (founder) → 3 (male), 4 (female)  ← full sibling
#   3 + 4 → 7 (male)
#   1 (founder) + 5 (founder) → 6 (female)             ← half sibling dengan 3&4

MOCK_PEDIGREE = {
    1: (None, None),
    2: (None, None),
    3: (1, 2),
    4: (1, 2),
    5: (None, None),
    6: (1, 5),
    7: (3, 4),
}


# ── Unit test

def test_coi_no_common_ancestor():
    """Domba tanpa common ancestor → COI harus 0"""
    coi = calculate_coi(2, 5, MOCK_PEDIGREE)
    assert coi == 0.0, f"Expected 0.0, got {coi}"

def test_coi_half_sibling():
    """Ewe(6) x Ram(3) punya common ancestor id=1 saja → COI 12.5%"""
    coi = calculate_coi(6, 3, MOCK_PEDIGREE)
    assert abs(coi - 0.125) < 0.001, f"Expected 0.125, got {coi}"

def test_coi_full_sibling():
    """Ewe(4) x Ram(3) full sibling → COI 25%"""
    coi = calculate_coi(4, 3, MOCK_PEDIGREE)
    assert abs(coi - 0.25) < 0.001, f"Expected 0.25, got {coi}"

def test_coi_founder():
    """Domba founder tanpa orang tua → COI 0%"""
    coi = calculate_coi(1, 2, MOCK_PEDIGREE)
    assert coi == 0.0, f"Expected 0.0, got {coi}"

def test_is_safe_pair_rejected():
    """Full sibling COI=25% harus ditolak"""
    is_safe, coi = is_safe_pair(4, 3, MOCK_PEDIGREE)
    assert is_safe == False
    assert abs(coi - 0.25) < 0.001

def test_is_safe_pair_accepted():
    """Tidak ada common ancestor → harus diterima"""
    is_safe, coi = is_safe_pair(2, 5, MOCK_PEDIGREE)
    assert is_safe == True
    assert coi == 0.0

def test_filter_safe_candidates():
    """Ram(3) full sibling dengan Ewe(4) → dibuang. Ram(7) juga punya CA → dibuang"""
    results = filter_safe_candidates(
        selected_id=4,
        candidates=[3, 7],
        pedigree=MOCK_PEDIGREE,
        is_ewe_selected=True
    )
    sheep_ids = [r["sheep_id"] for r in results]
    assert 3 not in sheep_ids, "Ram(3) harusnya dibuang (full sibling)"
    assert 7 not in sheep_ids, "Ram(7) harusnya dibuang (COI tinggi)"

def test_filter_returns_safe():
    """Ram(5) tidak ada CA dengan Ewe(4) → lolos"""
    results = filter_safe_candidates(
        selected_id=4,
        candidates=[5],
        pedigree=MOCK_PEDIGREE,
        is_ewe_selected=True
    )
    assert len(results) == 1
    assert results[0]["sheep_id"] == 5
    assert results[0]["coi"] == 0.0

def test_coi_offspring_of_full_siblings():
    """
    Ram(7) hasil dari full sibling (sire=3, dam=4)
    Pastikan rekursi menghitung F_A dengan benar
    """
    coi = calculate_coi(6, 7, MOCK_PEDIGREE)
    assert abs(coi - 0.125) < 0.001, f"Expected 0.125, got {coi}"

def test_coi_symmetry():
    """COI(A,B) harus sama dengan COI(B,A)"""
    coi_ab = calculate_coi(4, 3, MOCK_PEDIGREE)
    coi_ba = calculate_coi(3, 4, MOCK_PEDIGREE)
    assert coi_ab == coi_ba, f"COI tidak simetris: {coi_ab} vs {coi_ba}"

def test_memo_shared_across_candidates():
    """Pastikan memo benar-benar dipakai lintas kandidat"""
    memo = {}
    ancestor_cache = {}

    coi_1 = calculate_coi(4, 3, MOCK_PEDIGREE, _memo=memo, _ancestor_cache=ancestor_cache)
    size_after_first = len(memo)

    coi_2 = calculate_coi(4, 3, MOCK_PEDIGREE, _memo=memo, _ancestor_cache=ancestor_cache)
    size_after_second = len(memo)

    assert coi_1 == coi_2
    assert size_after_first == size_after_second, \
        "Memo bertambah padahal pasangan sama → cache tidak bekerja"

def test_filter_coi_percent_format():
    """Pastikan coi_percent konsisten dengan coi"""
    results = filter_safe_candidates(
        selected_id=4,
        candidates=[5],
        pedigree=MOCK_PEDIGREE,
        is_ewe_selected=True
    )
    for r in results:
        assert abs(r["coi_percent"] - r["coi"] * 100) < 0.001, \
            "coi_percent tidak konsisten dengan coi"

# ── Integration test

@pytest.mark.integration
def test_load_pedigree_from_db():
    """Pastikan pedigree ke-load dari DB dan tidak kosong"""
    pedigree = load_pedigree()
    assert len(pedigree) > 0, "Pedigree kosong — cek koneksi DB"
    # Semua value harus tuple of 2
    for sheep_id, parents in pedigree.items():
        assert isinstance(parents, tuple)
        assert len(parents) == 2

@pytest.mark.integration
def test_get_candidates_from_db():
    """Kandidat harus lawan jenis dan tidak termasuk dirinya sendiri"""
    candidates = get_candidates(selected_id=2, is_ewe_selected=True)
    assert isinstance(candidates, list)
    assert 2 not in candidates, "Domba tidak boleh jadi kandidat dirinya sendiri"
    assert len(candidates) > 0, "Harusnya ada kandidat Ram"

@pytest.mark.integration
def test_full_sibling_rejected_from_db():
    """Ewe(4) x Ram(3) full sibling dari DB → harus dibuang filter"""
    pedigree = load_pedigree()
    coi = calculate_coi(4, 3, pedigree)
    assert abs(coi - 0.25) < 0.001, f"Expected 25%, got {coi*100:.2f}%"
    is_safe, _ = is_safe_pair(4, 3, pedigree)
    assert is_safe == False
