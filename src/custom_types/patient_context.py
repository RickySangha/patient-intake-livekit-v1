from pydantic import BaseModel, Field
from typing import Optional, List, Union
import datetime  # Added for potential future use if needed directly here

# Import the new models and existing ones
from .complaint_details import ComplaintDetails
from .call_details import CallDetails
from .clinic import Clinic
from .doctor import Doctor
from .appointment import Appointment


class PatientContext(BaseModel):
    id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[datetime.datetime] = None
    main_issue: Optional[str] = None
    complaint_details: Optional[ComplaintDetails] = Field(None, discriminator="type")
    medical_history: Optional[dict] = None
    call_details: CallDetails = Field(default_factory=CallDetails)
    clinic: Optional[Clinic] = None
    doctor: Optional[Doctor] = None
    appointment: Optional[Appointment] = None
