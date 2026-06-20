from pydantic import BaseModel, Field


class WrightFilterRequest(BaseModel):
    selected_id    : int        = Field(..., description="ID domba yang dipilih user")
    candidate_ids  : list[int]  = Field(..., description="List ID kandidat lawan jenis")
    is_ewe_selected: bool       = Field(..., description="True jika yang dipilih betina")


class CandidateResult(BaseModel):
    sheep_id   : int
    coi        : float
    coi_percent: float


class WrightFilterResponse(BaseModel):
    safe_candidates: list[CandidateResult]
    total_input    : int
    total_safe     : int
    total_rejected : int
