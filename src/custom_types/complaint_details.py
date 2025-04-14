from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union


class BaseComplaintDetails(BaseModel):
    type: str


class ChestPainDetails(BaseComplaintDetails):
    type: Literal["chest_pain"] = "chest_pain"
    location: Optional[str] = None
    radiation: Optional[str] = None
    quality: Optional[str] = None
    severity: Optional[Union[int, str]] = None
    associated_symptoms: Optional[List[str]] = None
    aggravating_factors: Optional[List[str]] = None
    relieving_factors: Optional[List[str]] = None
    duration: Optional[str] = None
    onset: Optional[str] = None


class CoughDetails(BaseComplaintDetails):
    type: Literal["cough"] = "cough"
    duration: Optional[str] = None
    cough_type: Optional[str] = None
    triggers_patterns: Optional[List[str]] = None
    associated_symptoms: Optional[List[str]] = None
    severity_impact: Optional[str] = None
    is_productive: Optional[bool] = None
    sputum_color: Optional[str] = None


class BackPainDetails(BaseComplaintDetails):
    type: Literal["back_pain"] = "back_pain"
    location: Optional[str] = None
    quality: Optional[str] = None
    severity: Optional[Union[int, str]] = None
    duration: Optional[str] = None
    onset: Optional[str] = None
    aggravating_factors: Optional[List[str]] = None
    relieving_factors: Optional[List[str]] = None
    associated_symptoms: Optional[List[str]] = None
    radiation: Optional[str] = None


ComplaintDetails = Union[
    ChestPainDetails,
    CoughDetails,
    BackPainDetails,
]
