from pydantic import BaseModel
from typing import Optional, Dict, Any
import datetime


class CallDetails(BaseModel):
    call_id: Optional[str] = None
    consent_given: bool = False
    appointment_id: Optional[str] = None
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    call_status: Optional[str] = "scheduled"
    call_start_time: Optional[datetime.datetime] = None
    call_end_time: Optional[datetime.datetime] = None
    duration_minutes: Optional[float] = None
    collected_info: Optional[Dict[str, Any]] = None
    final_summary: Optional[str] = None
