"""
Wright Coefficient (Coefficient of Inbreeding / COI)
====================================================
Menghitung seberapa dekat kekerabatan dua domba (Ewe x Ram).
Hasil berupa float 0.0 - 1.0 (0% - 100%).

Threshold sistem: F >= 0.0625 (6.25%) -> pasangan ditolak.

Referensi:
  Wright, S. (1922). Coefficient of inbreeding and relationship.
  American Naturalist, 56, 330-338.
"""

from __future__ import annotations
import pandas as pd
from app.services.db import get_engine
from collections import deque

INBREEDING_THRESHOLD = 0.0625

# Silsilah

def load_pedigree() -> dict[int, tuple[int | None, int | None]]:
    """
    sire = bapak (ram), dam = ibu (ewe)
    """

    query = """
        SELECT id, sire_id, dam_id
        FROM sheep
        WHERE status = 'active'
    """

    df = pd.read_sql(query, get_engine())

    pedigree: dict[int, tuple[int | None, int | None]] = {}

    for _, row in df.iterrows():
        sire = int(row["sire_id"]) if pd.notna(row["sire_id"]) else None
        dam  = int(row["dam_id"])  if pd.notna(row["dam_id"])  else None
        pedigree[int(row["id"])] = (sire, dam)

    return pedigree

# Ancestor path

def get_ancestors(
    sheep_id: int,
    pedigree: dict[int, tuple[int | None, int | None]],
    max_gen: int = 6,
    _ancestor_cache: dict[tuple[int, int], dict[int, list[int]]] | None = None,
) -> dict[int, list[int]]:
    """
    Telusuri leluhur seekor domba ke atas hingga max_gen generasi.
    Return: { ancestor_id: [gen 1, gen 2, ...] }
    """

    if _ancestor_cache is not None:
        cache_key = (sheep_id, max_gen)
        if cache_key in _ancestor_cache:
            return _ancestor_cache[cache_key]

    ancestors: dict[int, list[int]] = {}
    queue = deque([(sheep_id, 0)])

    while queue:
        current_id, gen = queue.popleft()

        if gen > 0:
            if current_id not in ancestors:
                ancestors[current_id] = []
            ancestors[current_id].append(gen)

        if gen >= max_gen:
            continue

        if current_id not in pedigree:
            continue

        sire, dam = pedigree[current_id]

        if sire is not None:
            queue.append((sire, gen + 1))
        if dam is not None:
            queue.append((dam, gen + 1))

    if _ancestor_cache is not None:
        _ancestor_cache[cache_key] = ancestors

    return ancestors

# Wright Formula

def calculate_coi(
    ewe_id: int,
    ram_id: int,
    pedigree: dict[int, tuple[int | None, int | None]],
    max_gen: int = 6,
    _memo: dict[tuple[int, int, int], float] | None = None,
    _ancestor_cache: dict[tuple[int, int], dict[int, list[int]]] | None = None,
) -> float:

    if _memo is None:
        _memo = {}
    if _ancestor_cache is None:
        _ancestor_cache = {}

    if max_gen <= 0:
        return 0.0
    if ewe_id not in pedigree or ram_id not in pedigree:
        return 0.0

    sorted_key = (min(ewe_id, ram_id), max(ewe_id, ram_id), max_gen)
    if sorted_key in _memo:
        return _memo[sorted_key]

    ewe_ancestors = get_ancestors(ewe_id, pedigree, max_gen, _ancestor_cache)
    ram_ancestors = get_ancestors(ram_id, pedigree, max_gen, _ancestor_cache)

    common_ancestors = set(ewe_ancestors.keys()) & set(ram_ancestors.keys())

    if not common_ancestors:
        _memo[sorted_key] = 0.0
        return 0.0

    coi = 0.0
    for ca_id in common_ancestors:
        sire, dam = pedigree.get(ca_id, (None, None))

        if sire is None or dam is None:
            f_a = 0.0
        else:
            f_a = calculate_coi(dam, sire, pedigree, max_gen - 1, _memo, _ancestor_cache)

        for n1 in ewe_ancestors[ca_id]:
            for n2 in ram_ancestors[ca_id]:
                coi += (0.5 ** (n1 + n2 + 1)) * (1 + f_a)

    result = min(coi, 1.0)
    _memo[sorted_key] = result
    return round(result, 6)

def is_safe_pair(
    ewe_id: int,
    ram_id: int,
    pedigree: dict[int, tuple[int | None, int | None]] | None = None,
    threshold: float = INBREEDING_THRESHOLD,
    _memo: dict[tuple[int, int, int], float] | None = None,
    _ancestor_cache: dict[tuple[int, int], dict[int, list[int]]] | None = None,
) -> tuple[bool, float]:

    if pedigree is None:
        pedigree = load_pedigree()

    coi = calculate_coi(ewe_id, ram_id, pedigree, _memo=_memo, _ancestor_cache=_ancestor_cache)
    return (coi < threshold, coi)

def get_candidates(selected_id: int, is_ewe_selected: bool) -> list[int]:

    gender = "male" if is_ewe_selected else "female"

    query = """
        SELECT id FROM sheep
        WHERE status = 'active'
        AND gender = %s
        AND id != %s
    """

    df = pd.read_sql(query, get_engine(), params=(gender, selected_id))
    return df["id"].astype(int).tolist()

def filter_safe_candidates(
    selected_id: int,
    candidates: list[int],
    pedigree: dict[int, tuple[int | None, int | None]] | None = None,
    threshold: float = INBREEDING_THRESHOLD,
    is_ewe_selected: bool = True,
) -> list[dict]:

    if pedigree is None:
        pedigree = load_pedigree()

    memo           = {}
    ancestor_cache = {}

    results = []
    for candidate_id in candidates:
        if is_ewe_selected:
            ewe_id, ram_id = selected_id, candidate_id
        else:
            ewe_id, ram_id = candidate_id, selected_id

        is_safe, coi = is_safe_pair(
            ewe_id, ram_id, pedigree, threshold,
            _memo=memo,
            _ancestor_cache=ancestor_cache,
        )

        if is_safe:
            results.append({
                "sheep_id": candidate_id,
                "coi": coi,
                "coi_percent": round(coi * 100, 4)
            })

    return results
