from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime


class OpenSettings(BaseModel):
    agent_name: Optional[str] = None
    auto_call_enabled: Optional[bool] = False
    auto_call_settings: Optional[Dict[str, Any]] = (
        None  # Example: {"disabled_appointment_types": ["prescription_refill"]}
    )


class DoctorOnlySettings(BaseModel):
    rules: Optional[Dict[str, Any]] = (
        None  # Example: {"specific_patient_ids": ["patient789"]}
    )


class Clinic(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postalCode: Optional[str] = None
    phoneNumber: Optional[str] = None
    email: Optional[str] = None
    doctorIds: List[str] = Field(default_factory=list)
    patientIds: List[str] = Field(default_factory=list)
    open_settings: Optional[OpenSettings] = None
    doctor_only_settings: Optional[DoctorOnlySettings] = None
