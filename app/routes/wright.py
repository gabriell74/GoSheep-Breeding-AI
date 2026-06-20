"""
Route untuk endpoint Wright Coefficient filter.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.wright import (
    WrightFilterRequest,
    WrightFilterResponse,
    CandidateResult,
)
from app.services.wright_service import (
    load_pedigree,
    filter_safe_candidates,
)

router = APIRouter(prefix="/wright", tags=["Wright COI"])


@router.post("/filter", response_model=WrightFilterResponse)
def filter_candidates(request: WrightFilterRequest):
    """
    Filter kandidat breeding berdasarkan Wright COI.

    Alur:
        1. Load pedigree dari DB
        2. Filter kandidat yang COI >= 6.25% (ditolak)
        3. Return kandidat yang aman

    Dipanggil Laravel sebelum AHP + MOORA.
    """
    try:
        pedigree = load_pedigree()

        safe = filter_safe_candidates(
            selected_id     = request.selected_id,
            candidates      = request.candidate_ids,
            pedigree        = pedigree,
            is_ewe_selected = request.is_ewe_selected,
        )

        return WrightFilterResponse(
            safe_candidates = [CandidateResult(**c) for c in safe],
            total_input     = len(request.candidate_ids),
            total_safe      = len(safe),
            total_rejected  = len(request.candidate_ids) - len(safe),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
