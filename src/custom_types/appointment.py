from pydantic import BaseModel, Field
from typing import Optional
import datetime


class Appointment(BaseModel):
    id: Optional[str] = None
    patientId: Optional[str] = None
    doctorId: Optional[str] = None
    clinicId: Optional[str] = None
    appointmentType: Optional[str] = None
    appointmentDate: Optional[datetime.datetime] = None
    autoCallEnabled: Optional[bool] = False
    createdAt: Optional[datetime.datetime] = None
    updatedAt: Optional[datetime.datetime] = None
